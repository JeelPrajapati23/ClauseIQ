"""Step 1e: turn the reviewed data/test_to_review.csv into the frozen data/test.jsonl.

`human_label` blank means the reviewer agreed with `llm_label`; otherwise it overrides it.
Pass --reviewer to record who reviewed it. The current test set was reviewed by Claude
(claude-opus-5-5), not a human, at the project owner's request: it's a second,
independent model family checking gpt-oss-120b's labels against LABELLING_GUIDE.md,
which catches most labelling mistakes but is NOT human ground truth. Every row carries
`reviewed_by` so results can always say which kind of test set they were measured on.

Also collapses near-duplicates (_keys.near_dup_key: same normalised text prefix — e.g.
WildChat's many copies of one Midjourney prompt template), which exact-match dedupe in
fetch_sources.py misses and which would otherwise count one template many times over.
The training script drops train/val rows sharing a near_dup_key with any test row.

Usage: python Evaluation/offtopic/finalize_test_set.py --reviewer claude-opus-5-5
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from _keys import near_dup_key

DATA = Path(__file__).parent / "data"
LABELS = {"ON_TOPIC", "OFF_TOPIC"}
ALIASES = {"ON": "ON_TOPIC", "OFF": "OFF_TOPIC"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer", required=True, help='who reviewed the CSV, e.g. "human" or a model id')
    args = parser.parse_args()

    rows, overridden, bad, seen, dropped = [], [], [], set(), 0
    with (DATA / "test_to_review.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            human = r["human_label"].strip().upper()
            human = ALIASES.get(human, human)
            if human and human not in LABELS:
                bad.append((r["id"], r["human_label"]))
                continue
            key = near_dup_key(r["text"])
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            label = human or r["llm_label"]
            if human and human != r["llm_label"]:
                overridden.append(r)
            rows.append({"text": r["text"], "label": label, "llm_label": r["llm_label"],
                         "rule": r["llm_rule"], "unsure": r["llm_unsure"] == "True", "hard": r["hard"] == "True",
                         "source": r["source"], "seed_id": r["seed_id"], "reviewed_by": args.reviewer})
    if bad:
        raise SystemExit(f"Unrecognised human_label values (use ON_TOPIC/OFF_TOPIC or blank): {bad}")

    with (DATA / "test.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    labels = Counter(r["label"] for r in rows)
    print(f"test.jsonl: {len(rows)} rows  ON {labels['ON_TOPIC']}  OFF {labels['OFF_TOPIC']}  "
          f"(dropped {dropped} near-duplicates, reviewed by {args.reviewer})")
    print(f"reviewer overrode the LLM on {len(overridden)}/{len(rows)} rows ({len(overridden) / len(rows):.1%})")
    for r in overridden:
        print(f"  {r['llm_label']} -> {r['human_label'].strip().upper()}  [{r['source']}] {r['text'][:100]}")


if __name__ == "__main__":
    main()
