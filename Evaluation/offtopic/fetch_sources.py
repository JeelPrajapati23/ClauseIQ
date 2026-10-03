"""Step 1a of the off-topic classifier dataset: pull candidate questions from public
datasets into data/raw/<source>.jsonl. See LABELLING_GUIDE.md for how they get labelled.

Every row carries a `prior` — the label its source suggests — which the LLM labeller
(label_with_llm.py) does NOT see; a row whose final label disagrees with its prior is
flagged for human review. Sources verified live 2026-10-02 (licence, size, sample rows):

  privacyqa        ON prior   MIT          github.com/AbhilashaRavichander/PrivacyQA_EMNLP
  consumer_contracts ON prior CC-BY-NC-4.0 mteb/legalbench_consumer_contracts_qa (drop if ever commercial)
  law_se           ON prior   CC-BY-SA-4.0 ymoslem/Law-StackExchange (grey zone, ON-5/ON-6)
  clinc            OFF prior  CC-BY-3.0    clinc/clinc_oos (plus), fetched from github.com/clinc/oos-eval
  wildchat         no prior   ODC-BY       allenai/WildChat-1M (English, non-toxic, first user turn)

Uses plain HTTP (HF datasets-server rows API + GitHub raw) rather than the `datasets`
library, so it runs on Windows without pyarrow (see CLAUDE.md's Smart App Control note).
Usage: python Evaluation/offtopic/fetch_sources.py
"""
import csv
import hashlib
import io
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).parent / "data" / "raw"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PRIVACYQA_URL = "https://raw.githubusercontent.com/AbhilashaRavichander/PrivacyQA_EMNLP/master/data/{}.csv"
# Single-file copy of the same clinc/clinc_oos "plus" config — paging ~240 requests
# through the rows API trips its rate limit.
CLINC_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_oos_plus.json"

# Rows kept per source before labelling — sized so OFF_TOPIC doesn't swamp the
# ~2k natural ON_TOPIC questions available (see the dataset verification notes).
N_CLINC = 1500
N_WILDCHAT = 2500
N_LAW_SE = 1000
MAX_CHARS = 2000  # prod caps questions at 4000; long pasted prompts are trimmed, not dropped

random.seed(42)


def _get_json(url: str, retries: int = 6) -> dict:
    for attempt in range(retries):
        time.sleep(0.5)  # stay well under the datasets-server rate limit
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if attempt == retries - 1:
                raise
            time.sleep(15 * 2 ** attempt if e.code == 429 else 2 ** attempt)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _rows(dataset: str, config: str, split: str, offset: int, length: int = 100) -> tuple[int, list[dict]]:
    qs = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                 "offset": offset, "length": length})
    j = _get_json(f"{ROWS_API}?{qs}")
    return j["num_rows_total"], [r["row"] for r in j["rows"]]


def _all_rows(dataset: str, config: str, split: str) -> list[dict]:
    total, out = None, []
    while total is None or len(out) < total:
        total, rows = _rows(dataset, config, split, len(out))
        if not rows:
            break
        out.extend(rows)
    return out


def _record(text: str, source: str, prior: str | None, seed_key: str) -> dict:
    text = " ".join(text.split())[:MAX_CHARS]
    return {
        "text": text,
        "source": source,
        "prior": prior,
        "seed_id": f"{source}-{hashlib.sha1(seed_key.encode()).hexdigest()[:12]}",
    }


def _dedupe(records: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in records:
        key = r["text"].lower().strip(" ?.!")
        if r["text"] and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def fetch_privacyqa() -> list[dict]:
    out = []
    for name in ("policy_train_data", "policy_test_data"):
        with urllib.request.urlopen(PRIVACYQA_URL.format(name), timeout=300) as resp:
            reader = csv.DictReader(io.TextIOWrapper(resp, encoding="utf-8"), delimiter="\t")
            for row in reader:
                out.append(_record(row["Query"], "privacyqa", "ON_TOPIC", row["Query"].lower()))
    return out


def fetch_consumer_contracts() -> list[dict]:
    rows = _all_rows("mteb/legalbench_consumer_contracts_qa", "queries", "queries")
    return [_record(r["text"], "consumer_contracts", "ON_TOPIC", r["_id"]) for r in rows]


def fetch_clinc() -> list[dict]:
    data = _get_json(CLINC_URL)
    # Stratify across all 151 intents (150 + "oos") so no single assistant domain dominates.
    by_intent: dict[str, list[str]] = {}
    for split_rows in data.values():
        for text, intent in split_rows:
            by_intent.setdefault(intent, []).append(text)
    per_intent = max(1, N_CLINC // len(by_intent))
    picked = []
    for group in by_intent.values():
        picked += random.sample(group, min(per_intent, len(group)))
    return [_record(t, "clinc", "OFF_TOPIC", t) for t in picked]


def fetch_wildchat() -> list[dict]:
    total, _ = _rows("allenai/WildChat-1M", "default", "train", 0, 1)
    out = []
    while len(out) < N_WILDCHAT:
        _, rows = _rows("allenai/WildChat-1M", "default", "train", random.randrange(total - 100))
        for r in rows:
            if r["language"] != "English" or r["toxic"]:
                continue
            first_user = next((m for m in r["conversation"] if m["role"] == "user"), None)
            if first_user and first_user["content"].strip():
                out.append(_record(first_user["content"], "wildchat", None, r["conversation_hash"]))
        out = _dedupe(out)
        print(f"  wildchat: {len(out)}/{N_WILDCHAT}")
    return out[:N_WILDCHAT]


def fetch_law_se() -> list[dict]:
    total, _ = _rows("ymoslem/Law-StackExchange", "default", "train", 0, 1)
    out = []
    for offset in random.sample(range(0, total - 100, 100), N_LAW_SE // 100 + 2):
        _, rows = _rows("ymoslem/Law-StackExchange", "default", "train", offset)
        out += [_record(r["question_title"], "law_se", "ON_TOPIC", str(r["question_id"]))
                for r in rows if r.get("question_title")]
    return _dedupe(out)[:N_LAW_SE]


SOURCES = {
    "privacyqa": fetch_privacyqa,
    "consumer_contracts": fetch_consumer_contracts,
    "clinc": fetch_clinc,
    "wildchat": fetch_wildchat,
    "law_se": fetch_law_se,
}


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, fetch in SOURCES.items():
        path = RAW_DIR / f"{name}.jsonl"
        if path.exists():
            print(f"{name}: already fetched, skipping ({path})")
            continue
        print(f"{name}: fetching...")
        records = _dedupe(fetch())
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(records)} rows -> {path}")


if __name__ == "__main__":
    main()
