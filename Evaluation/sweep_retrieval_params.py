"""
Fast, judge-free sweep of retrieval parameters in app.database.get_reranking_retriever,
against the golden set's own ground_truth text.

Unlike the full evaluate_rag_offline.py run (real generation + a Ragas judge, much
slower), this only runs retrieval and does a deterministic substring check — no LLM
calls at all — so it can sweep several settings quickly. Use it to pick a setting, then
confirm with one real evaluate_rag_offline.py run before treating it as final.

Sweeps initial_k / final_k — the candidate pool size and final chunk count each
retriever contributes before/after reranking.

Usage:
    python Evaluation/sweep_retrieval_params.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from app.database import get_reranking_retriever  # noqa: E402
from app.generator import QueryIntent, classify_intent  # noqa: E402

GOLDEN_SET_FILE = Path(__file__).resolve().parent / "golden_qa_set.json"
GOLDEN_DOC_MAP_FILE = Path(__file__).resolve().parent / "golden_doc_map.json"
EVAL_COLLECTION_NAME = "eval_knowledge_base"
EVAL_USER_ID = "eval"

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()


def is_hit(ground_truth: str, retrieved_docs: list) -> bool:
    """Shrinking-probe substring match against ground_truth (the short answer text) —
    NOT reference_contexts, which for this golden set holds the *entire* source
    document (CUAD-style full-document context, not a targeted excerpt) rather than a
    specific relevant snippet; matching against it would check almost nothing
    meaningful. Same shrinking-probe approach as build_golden_doc_map.py's
    find_matches, tolerant of chunk-boundary/reflow differences."""
    normalized_gt = normalize(ground_truth)
    retrieved_text = normalize(" ".join(d.page_content for d in retrieved_docs))
    for length in (200, 120, 80, 50):
        probe = normalized_gt[:length]
        if len(probe) < 30:
            continue
        if probe in retrieved_text:
            return True
    return False


def load_rows() -> list[dict]:
    golden = json.loads(GOLDEN_SET_FILE.read_text(encoding="utf-8"))
    doc_map = json.loads(GOLDEN_DOC_MAP_FILE.read_text(encoding="utf-8"))
    rows = []
    for i, row in enumerate(golden):
        entry = doc_map.get(str(i), {})
        source_file = entry.get("source_file")
        ground_truth = row.get("ground_truth") or ""
        if not source_file or not ground_truth.strip():
            continue  # unscoped or guardrail rows have no single-document target to check
        rows.append({
            "source_row": i,
            "question": row["question"],
            "ground_truth": ground_truth,
            "source_file": source_file,
            "intent": row.get("intent", "FACTUAL"),
        })
    return rows


def sweep_initial_k(k_values: list[int]) -> None:
    """final_k stays fixed at the current production values (3 FACT / 5 ANALYTICAL) so
    this isolates the effect of candidate-pool size on whether the reranker even gets a
    chance to surface the right chunk, rather than conflating it with a final_k change."""
    rows = load_rows()
    print(f"Sweeping {len(k_values)} initial_k values over {len(rows)} scoped golden rows.\n")

    for k in k_values:
        hits = 0
        hits_by_intent: dict[str, list[int]] = {}
        for row in rows:
            intent = classify_intent(row["question"])
            retriever = get_reranking_retriever(
                user_id=EVAL_USER_ID,
                collection_name=EVAL_COLLECTION_NAME,
                document_filter=[row["source_file"]],
                initial_k=k,
                final_k=5 if intent == QueryIntent.ANALYTICAL else 3,
            )
            retrieved = retriever.invoke(row["question"])
            time.sleep(6.5)  # stay under the Cohere rate limit
            hit = is_hit(row["ground_truth"], retrieved)
            hits += int(hit)
            hits_by_intent.setdefault(row["intent"], []).append(int(hit))

        rate = hits / len(rows)
        print(f"initial_k={k}: hit-rate={rate:.3f} ({hits}/{len(rows)})")
        for intent, results in sorted(hits_by_intent.items()):
            sub_rate = sum(results) / len(results)
            print(f"    {intent}: {sub_rate:.3f} ({sum(results)}/{len(results)})")
    print()


def sweep_final_k(k_values: list[int]) -> None:
    """initial_k stays fixed at 20 (the production value) so this isolates the effect
    of how many reranked candidates reach the LLM, independent of pool size."""
    rows = load_rows()
    print(f"Sweeping {len(k_values)} final_k values over {len(rows)} scoped golden rows (initial_k=20).\n")

    for k in k_values:
        hits = 0
        hits_by_intent: dict[str, list[int]] = {}
        for row in rows:
            retriever = get_reranking_retriever(
                user_id=EVAL_USER_ID,
                collection_name=EVAL_COLLECTION_NAME,
                document_filter=[row["source_file"]],
                initial_k=20,
                final_k=k,
            )
            retrieved = retriever.invoke(row["question"])
            time.sleep(6.5)  # stay under the Cohere rate limit
            hit = is_hit(row["ground_truth"], retrieved)
            hits += int(hit)
            hits_by_intent.setdefault(row["intent"], []).append(int(hit))

        rate = hits / len(rows)
        print(f"final_k={k}: hit-rate={rate:.3f} ({hits}/{len(rows)})")
        for intent, results in sorted(hits_by_intent.items()):
            sub_rate = sum(results) / len(results)
            print(f"    {intent}: {sub_rate:.3f} ({sum(results)}/{len(results)})")
    print()


if __name__ == "__main__":
    sweep_final_k([3, 4, 5, 6, 8, 10])
