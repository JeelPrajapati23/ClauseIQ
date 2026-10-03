"""Latency of the old off-topic gate (one Gemini call per question) vs the new one
(embedding + numpy scoring, plus a Gemini call only for the uncertain band), on the same
100-question sample compare_classifiers.py used.

Old-gate latencies are the Gemini call times compare_classifiers.py already recorded in
results/compare_gemini_cache.jsonl, so this spends no Gemini quota. New-gate latencies are
measured live here: one uncached HF embedding call per question (free) and the numpy
scoring step timed separately; questions the classifier defers to Gemini add their
recorded hardened-prompt Gemini time. Both are measured from the same machine, so the
comparison is fair even though absolute numbers differ from Render's network position.

Usage (project venv, repo root): python Evaluation/offtopic/benchmark_latency.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from _keys import text_key  # noqa: E402
from app.database import embeddings  # noqa: E402
from app.offtopic_classifier import load_classifier  # noqa: E402

RESULTS = Path(__file__).parent / "results"
QUERY_PREFIX = "query: "  # same as _ArcticLegalEmbeddings.embed_query, but bypassing its cache


def pct(xs: list[float], q: float) -> float:
    return round(float(np.percentile(xs, q)), 1)


def summary(xs: list[float]) -> dict:
    return {"n": len(xs), "mean": round(float(np.mean(xs)), 1), "p50": pct(xs, 50),
            "p90": pct(xs, 90), "p99": pct(xs, 99), "max": round(float(max(xs)), 1)}


def main():
    sample = [json.loads(line) for line in (RESULTS / "compare_sample.jsonl").open(encoding="utf-8")]
    cache = {(e["prompt"], e["key"]): e for e in map(json.loads, (RESULTS / "compare_gemini_cache.jsonl").open(encoding="utf-8"))}
    clf = load_classifier()

    embeddings.embed_documents([QUERY_PREFIX + "warm-up"])  # exclude connection setup from row 1
    old_ms, embed_ms, score_ms, new_ms, decisions = [], [], [], [], []
    for r in sample:
        k = text_key(r["text"])
        old_ms.append(cache[("old", k)]["latency_ms"])

        t = time.perf_counter()
        vec = embeddings.embed_documents([QUERY_PREFIX + r["text"]])[0]
        e = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        for _ in range(1000):
            p = clf.p_off(vec)
        s = (time.perf_counter() - t) * 1000 / 1000
        d = clf.decide(p)

        total = e + s + (cache[("new", k)]["latency_ms"] if d == "defer" else 0)
        embed_ms.append(e); score_ms.append(s); new_ms.append(total); decisions.append(d)

    decided = [n for n, d in zip(new_ms, decisions) if d != "defer"]
    report = {
        "old_gate_gemini_ms": summary(old_ms),
        "new_gate_total_ms": summary(new_ms),
        "new_gate_embedding_ms": summary(embed_ms),
        "new_gate_scoring_ms": summary(score_ms),
        "new_gate_decided_without_gemini_ms": summary(decided),
        "deferred_to_gemini": decisions.count("defer"),
        "speedup_mean": round(float(np.mean(old_ms) / np.mean(new_ms)), 1),
        "speedup_p50": round(float(np.median(old_ms) / np.median(new_ms)), 1),
    }
    (RESULTS / "latency_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, v in report.items():
        print(f"{name:38} {v}")


if __name__ == "__main__":
    main()
