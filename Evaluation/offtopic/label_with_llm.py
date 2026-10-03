"""Step 1c of the off-topic classifier dataset: label every row in data/raw/*.jsonl
with gpt-oss-120b, following LABELLING_GUIDE.md, into data/labelled.jsonl.

The labeller sees only the text — never the row's source or `prior` (the label its
source/generation intended) — so the two are independent opinions. A row gets
`needs_review: true` when they disagree or the labeller marks it unsure; those rows are
what a human checks. The rules are read straight from LABELLING_GUIDE.md (core test
through tie-breakers), so the guide stays the single source of truth.

Resumable: rows whose text is already in the output are skipped.
Usage: python Evaluation/offtopic/label_with_llm.py   (from the project venv)
       python Evaluation/offtopic/label_with_llm.py --relabel-review
           re-labels only rows flagged needs_review, against the current guide — run after
           a guide change, so agreed rows keep their labels and only disputes are redone
"""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _groq import chat_json, n_keys  # noqa: E402

HERE = Path(__file__).parent
RAW_DIR = HERE / "data" / "raw"
OUT_PATH = HERE / "data" / "labelled.jsonl"
GUIDE_PATH = HERE / "LABELLING_GUIDE.md"

BATCH_SIZE = 20
LABELS = {"ON_TOPIC", "OFF_TOPIC"}


def _guide_rules() -> str:
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    return guide[guide.index("## The core test"):guide.index("## Record format")].strip()


PROMPT = """You are labelling inputs for a classifier that guards ClauseIQ, a legal-document Q&A app. Each input is something a user typed into the app's question box. Label each one by these rules:

{rules}

Inputs:
{items}

Return JSON: {{"labels": [{{"id": <input id>, "label": "ON_TOPIC" or "OFF_TOPIC", "rule": "<the deciding rule id, e.g. ON-5 or OFF-2>", "unsure": true/false}}]}} — exactly one entry per input id."""


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _clip(text: str) -> str:
    # Long rows are pasted text; keep both ends so a trailing question survives.
    return text if len(text) <= 700 else f"{text[:450]} […] {text[-250:]}"


def label_batch(rows: list[dict]) -> list[dict]:
    items = "\n".join(f"[{i}] {json.dumps(_clip(r['text']), ensure_ascii=False)}" for i, r in enumerate(rows))
    for _ in range(2):
        out = chat_json(PROMPT.format(rules=RULES, items=items), max_tokens=3000)
        by_id = {e.get("id"): e for e in out.get("labels", []) if e.get("label") in LABELS}
        if set(by_id) == set(range(len(rows))):
            break
    else:
        raise ValueError(f"labeller returned {len(by_id)}/{len(rows)} valid labels")
    labelled = []
    for i, r in enumerate(rows):
        e = by_id[i]
        unsure = bool(e.get("unsure"))
        labelled.append({
            **r,
            "label": e["label"],
            "rule": str(e.get("rule", "")),
            "unsure": unsure,
            "needs_review": unsure or (r.get("prior") is not None and r["prior"] != e["label"]),
        })
    return labelled


RULES = _guide_rules()


def relabel_review():
    rows = [json.loads(line) for line in OUT_PATH.open(encoding="utf-8")]
    todo = [i for i, r in enumerate(rows) if r["needs_review"]]
    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    print(f"re-labelling {len(todo)} needs_review rows in {len(batches)} batches")
    strip = ("label", "rule", "unsure", "needs_review")
    with ThreadPoolExecutor(max_workers=n_keys()) as pool:
        futures = {pool.submit(label_batch, [{k: v for k, v in rows[i].items() if k not in strip} for i in b]): b
                   for b in batches}
        for k, fut in enumerate(as_completed(futures), 1):
            for i, new in zip(futures[fut], fut.result()):
                rows[i] = {**new, "relabelled": True}
            if k % 10 == 0:
                print(f"  {k}/{len(batches)} batches")
    tmp = OUT_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(OUT_PATH)
    print(f"done; still needs_review: {sum(r['needs_review'] for r in rows)}")


def main():
    if "--relabel-review" in sys.argv:
        return relabel_review()
    done = set()
    if OUT_PATH.exists():
        done = {_key(json.loads(line)["text"]) for line in OUT_PATH.open(encoding="utf-8")}
    todo = []
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        todo += [r for r in map(json.loads, path.open(encoding="utf-8")) if _key(r["text"]) not in done]
    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    print(f"{len(done)} already labelled, {len(todo)} to label in {len(batches)} batches on {n_keys()} keys")

    failed = 0
    with ThreadPoolExecutor(max_workers=n_keys()) as pool, OUT_PATH.open("a", encoding="utf-8") as f:
        futures = [pool.submit(label_batch, b) for b in batches]
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                rows = fut.result()
            except Exception as exc:  # rerun resumes failed batches
                failed += 1
                print(f"  batch failed: {exc!r}")
                continue
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if k % 25 == 0:
                print(f"  {k}/{len(batches)} batches")
    print(f"done ({failed} batches failed — rerun to retry them)")


if __name__ == "__main__":
    main()
