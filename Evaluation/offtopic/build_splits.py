"""Step 1d of the off-topic classifier dataset: split data/labelled.jsonl into
train / val / test, grouped by seed_id so paraphrases of one seed never straddle splits.

  data/train.jsonl, data/val.jsonl   LLM labels (gpt-oss-120b, per LABELLING_GUIDE.md)
  data/test_to_review.csv            the frozen test set, ~400 rows, for a HUMAN to check:
                                     fill in `human_label` (ON_TOPIC / OFF_TOPIC) on every
                                     row — blank means "agree with llm_label" — then run
                                     finalize_test_set.py to produce data/test.jsonl

Every seed_id is assigned to test / val / train by a stable hash, so rerunning gives the
same split. Only a stratified sample of the test pool goes to human review; the rest of
the test pool is dropped rather than moved into train, which keeps the grouping intact.
Test quotas lean ON_TOPIC (~60%) to roughly reflect real traffic; on top of the quotas,
every *hard* row in the test pool is included (see _is_hard), since a random sample is
mostly easy cases and those alone can't tell topic-understanding from style-matching.
golden_set rows are never used anywhere, so the RAG eval's guardrail rows stay an
independent check. generated_targeted rows (categories picked from test errors) only ever
go to train/val. Once data/test.jsonl exists the test set is frozen: rerunning rewrites
train/val but leaves test_to_review.csv alone, so a reviewed CSV is never resampled.

Usage: python Evaluation/offtopic/build_splits.py
"""
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
LABELLED = DATA / "labelled.jsonl"

TEST_FRAC, VAL_FRAC = 0.15, 0.15
# Rows drawn from the test pool per source for human review (~400 total).
TEST_QUOTA = {
    "generated": 130, "wildchat": 80, "privacyqa": 50, "law_se": 50,
    "clinc": 50, "consumer_contracts": 25, "chat_message": 10_000,  # every real question, if any
}

random.seed(11)


TRAIN_VAL_ONLY_SOURCES = {"generated_targeted"}


def _split_of(seed_id: str, source: str) -> str:
    h = int(hashlib.sha1(seed_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if source in TRAIN_VAL_ONLY_SOURCES:
        return "val" if h < VAL_FRAC / (1 - TEST_FRAC) else "train"  # same train:val ratio as other sources
    return "test" if h < TEST_FRAC else "val" if h < TEST_FRAC + VAL_FRAC else "train"


def _is_hard(r: dict) -> bool:
    """Rows whose label goes against their source's usual label or surface style."""
    return (
        (r["source"] == "generated" and r["label"] == "OFF_TOPIC")  # generated hard negatives
        or (r["source"] in ("wildchat", "clinc") and r["label"] == "ON_TOPIC")
        or (r["source"] in ("privacyqa", "law_se", "consumer_contracts") and r["label"] == "OFF_TOPIC")
        or r["needs_review"]
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    rows = [json.loads(line) for line in LABELLED.open(encoding="utf-8")]
    rows = [r for r in rows if r["source"] != "golden_set"]
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in rows:
        splits[_split_of(r["seed_id"], r["source"])].append(r)

    test = [r for r in splits["test"] if _is_hard(r)]
    for source, quota in TEST_QUOTA.items():
        pool = [r for r in splits["test"] if r["source"] == source and not _is_hard(r)]
        test += random.sample(pool, min(quota, len(pool)))
    random.shuffle(test)

    _write_jsonl(DATA / "train.jsonl", splits["train"])
    _write_jsonl(DATA / "val.jsonl", splits["val"])
    if (DATA / "test.jsonl").exists():
        print("test.jsonl exists: test set is frozen, test_to_review.csv left untouched")
        test = [json.loads(line) for line in (DATA / "test.jsonl").open(encoding="utf-8")]
    else:
        _write_review_csv(test)

    for name, rs in (("train", splits["train"]), ("val", splits["val"]), ("test", test)):
        labels = Counter(r["label"] for r in rs)
        print(f"{name:17} {len(rs):5} rows  ON {labels['ON_TOPIC']:5}  OFF {labels['OFF_TOPIC']:5}  "
              f"needs_review {sum(r.get('needs_review', False) for r in rs)}")
    print("by source (train):", dict(Counter((r["source"], r["label"]) for r in splits["train"])))


def _write_review_csv(test: list[dict]) -> None:
    with (DATA / "test_to_review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "llm_label", "llm_rule", "llm_unsure", "hard", "human_label", "source", "seed_id"])
        for i, r in enumerate(test):
            w.writerow([i, r["text"], r["label"], r["rule"], r["unsure"], _is_hard(r), "", r["source"], r["seed_id"]])


if __name__ == "__main__":
    main()
