"""
Fast head-to-head comparison of two candidate generation models on real retrieved
context from the eval corpus (Qdrant's eval_knowledge_base collection).

This is NOT a replacement for a full Evaluation/evaluate_rag_offline.py Ragas run. It
only proxies faithfulness, using the pipeline's own two-call claim verifier
(app.generator.verify_answer_claims, fixed on the same model for both candidates so
the comparison is apples-to-apples). It does not measure answer_relevancy,
context_precision, or context_recall — those need the real Ragas run. Use this for
fast directional signal before spending the time on that.

Retrieval is shared per row (context doesn't depend on which model answers), and both
candidates are tested against the same prompt pair so this isolates model choice, not
prompt choice.

Requires: Qdrant running with eval_knowledge_base populated (see ingest_eval_corpus.py),
GROQ_API_KEY (generation + verifier), COHERE_API_KEY (reranker), and a key for whatever
second candidate provider is being compared.

Usage:
    python Evaluation/compare_generation_models.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Test both candidates against the same prompt pair, independent of whatever the
# current default env vars point to.
os.environ.setdefault("RAG_SYSTEM_PROMPT_FILE", "system_prompt_v4.txt")
os.environ.setdefault("RAG_ANALYTICAL_PROMPT_FILE", "system_prompt_analytical_v2.txt")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from groq import RateLimitError  # noqa: E402

from app.database import get_full_document_context, get_reranking_retriever  # noqa: E402
from app.generator import (  # noqa: E402
    QueryIntent,
    classify_intent,
    load_prompt,
    needs_full_document_read,
    stream_answer,
    verify_answer_claims,
)
from gemini_rotation import RotatingGeminiClient, load_gemini_keys  # noqa: E402

EVAL_COLLECTION_NAME = "eval_knowledge_base"
EVAL_USER_ID = "eval"
MAX_FULL_DOC_CHARS = 20000
GOLDEN_SET_FILE = Path(__file__).resolve().parent / "golden_qa_set.json"
GOLDEN_DOC_MAP_FILE = Path(__file__).resolve().parent / "golden_doc_map.json"
OUT_FILE = Path(__file__).resolve().parent / "model_compare_results.json"

GEMINI_MODEL = os.getenv("GEMINI_COMPARE_MODEL", "gemini-2.5-flash")
_GEMINI_FACT_PROMPT = "system_prompt_v4.txt"
_GEMINI_ANALYTICAL_PROMPT = "system_prompt_analytical_v2.txt"

_gemini_keys = load_gemini_keys()
if not _gemini_keys:
    raise RuntimeError("Set GEMINI_API_KEY (comma-separated for multiple) before running this.")
_gemini_client = RotatingGeminiClient(GEMINI_MODEL, _gemini_keys)
if len(_gemini_keys) > 1:
    print(f"Gemini key rotation active: {len(_gemini_keys)} keys.\n")


def gemini_generate(question: str, formatted_context: str, intent: QueryIntent) -> str:
    """Mirrors app.generator.stream_answer's prompt selection, but calls Gemini (via the
    rotating client above) instead of the Groq-backed llm."""
    prompt_file = _GEMINI_ANALYTICAL_PROMPT if intent == QueryIntent.ANALYTICAL else _GEMINI_FACT_PROMPT
    system_prompt = load_prompt(prompt_file).replace("{context}", formatted_context)
    return _gemini_client.generate(system_prompt, question)


def _format_page_range(pages: list) -> str:
    """Merge consecutive page numbers into compact ranges. Mirrors app.main."""
    if not pages:
        return ""
    try:
        nums = sorted(set(int(p) for p in pages))
    except (TypeError, ValueError):
        return ", ".join(str(p) for p in pages)
    ranges, start, end = [], nums[0], nums[0]
    for p in nums[1:]:
        if p == end + 1:
            end = p
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = p
    ranges.append(f"{start}-{end}" if start != end else str(start))
    prefix = "pp." if len(nums) > 1 else "p."
    return f"{prefix} {', '.join(ranges)}"


def load_rows(n_factual: int = 12, n_analytical: int = 8) -> list[dict]:
    """Same scoped-row filter as sweep_retrieval_params.py, evenly sampled across the
    two intents so the small sample still covers both."""
    golden = json.loads(GOLDEN_SET_FILE.read_text(encoding="utf-8"))
    doc_map = json.loads(GOLDEN_DOC_MAP_FILE.read_text(encoding="utf-8"))
    factual, analytical = [], []
    for i, row in enumerate(golden):
        entry = doc_map.get(str(i), {})
        source_file = entry.get("source_file")
        ground_truth = (row.get("ground_truth") or "").strip()
        if not source_file or not ground_truth:
            continue
        item = {
            "source_row": i,
            "question": row["question"],
            "source_file": source_file,
            "intent": row.get("intent", "FACTUAL"),
        }
        (analytical if item["intent"] == "ANALYTICAL" else factual).append(item)

    def sample(lst: list, n: int) -> list:
        if len(lst) <= n:
            return lst
        step = len(lst) / n
        return [lst[int(i * step)] for i in range(n)]

    return sample(factual, n_factual) + sample(analytical, n_analytical)


def retrieve_context(question: str, source_file: str):
    intent = classify_intent(question)
    full_doc_docs = (
        get_full_document_context(user_id=EVAL_USER_ID, source_file=source_file, collection_name=EVAL_COLLECTION_NAME)
        if needs_full_document_read(question)
        else None
    )
    if full_doc_docs and sum(len(d.page_content) for d in full_doc_docs) <= MAX_FULL_DOC_CHARS:
        retrieved_docs = full_doc_docs
    else:
        retriever = get_reranking_retriever(
            user_id=EVAL_USER_ID,
            collection_name=EVAL_COLLECTION_NAME,
            initial_k=20,
            final_k=5 if intent == QueryIntent.ANALYTICAL else 6,
            document_filter=[source_file],
        )
        retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return intent, None

    context_parts = []
    for doc in retrieved_docs:
        sf = doc.metadata.get("source_file", "Unknown")
        all_pages = doc.metadata.get("all_pages") or [doc.metadata.get("page", "?")]
        section = doc.metadata.get("section", "")
        header = f"Source: {sf}, Pages: {_format_page_range(all_pages)}"
        if section:
            header += f", Section: {section}"
        context_parts.append(f"{header}\nContent: {doc.page_content}\n")
    formatted_context = "\n---\n".join(context_parts)
    return intent, formatted_context


def main():
    rows = load_rows()
    n_fact = sum(1 for r in rows if r["intent"] == "FACTUAL")
    n_ana = sum(1 for r in rows if r["intent"] == "ANALYTICAL")
    print(f"Comparing qwen/qwen3.8-27b vs {GEMINI_MODEL} on {len(rows)} rows "
          f"({n_fact} FACTUAL / {n_ana} ANALYTICAL)\n")

    if OUT_FILE.exists():
        results = json.loads(OUT_FILE.read_text(encoding="utf-8"))
        results.setdefault("qwen", [])
        results.setdefault("gemini", [])
    else:
        results = {"qwen": [], "gemini": []}
    done_rows = {r["row"] for r in results["qwen"]} & {r["row"] for r in results["gemini"]}
    if done_rows:
        print(f"Resuming — {len(done_rows)} rows already completed, skipping those.\n")

    for i, row in enumerate(rows):
        if row["source_row"] in done_rows:
            continue
        print(f"[{i + 1}/{len(rows)}] row {row['source_row']} ({row['intent']}): {row['question'][:70]}")
        intent, formatted_context = retrieve_context(row["question"], row["source_file"])
        time.sleep(6.5)  # stay under the Cohere rate limit

        if formatted_context is None:
            print("    no context retrieved, skipping")
            continue

        try:
            qwen_answer = "".join(stream_answer(row["question"], formatted_context, history=[], intent=intent))
            qwen_report = verify_answer_claims(row["question"], formatted_context, qwen_answer)
            results["qwen"].append({
                "row": row["source_row"], "intent": row["intent"],
                "score": qwen_report.faithfulness_score, "verdict": qwen_report.verdict,
                "answer": qwen_answer,
            })
            print(f"    qwen:   {qwen_report.verdict:8s} {qwen_report.faithfulness_score:.2f}")
        except RateLimitError as exc:
            # stream_answer's own retry/backoff already failed before this propagated —
            # record it as a real reliability data point rather than crashing the run.
            results["qwen"].append({
                "row": row["source_row"], "intent": row["intent"],
                "score": 0.0, "verdict": "ERROR", "answer": "", "error": str(exc)[:300],
            })
            print(f"    qwen:   ERROR    ({str(exc)[:120]})")

        try:
            gemini_answer = gemini_generate(row["question"], formatted_context, intent)
            gemini_report = verify_answer_claims(row["question"], formatted_context, gemini_answer)
            results["gemini"].append({
                "row": row["source_row"], "intent": row["intent"],
                "score": gemini_report.faithfulness_score, "verdict": gemini_report.verdict,
                "answer": gemini_answer,
            })
            print(f"    gemini: {gemini_report.verdict:8s} {gemini_report.faithfulness_score:.2f}")
        except (ServerError, ClientError) as exc:
            results["gemini"].append({
                "row": row["source_row"], "intent": row["intent"],
                "score": 0.0, "verdict": "ERROR", "answer": "", "error": str(exc)[:300],
            })
            print(f"    gemini: ERROR    ({str(exc)[:120]})")

        # Write after every row so a mid-run failure (e.g. a provider outage) doesn't
        # discard already-completed rows — same resumability rationale as
        # evaluate_rag_offline.py's CSV-append behavior.
        OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n--- Summary ---")
    for name, rows_ in results.items():
        if not rows_:
            continue
        avg = sum(r["score"] for r in rows_) / len(rows_)
        verdicts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
        for r in rows_:
            verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
        print(f"{name}: avg faithfulness_score={avg:.3f}  verdicts={verdicts}")
        for intent_label in ("FACTUAL", "ANALYTICAL"):
            sub = [r for r in rows_ if r["intent"] == intent_label]
            if sub:
                sub_avg = sum(r["score"] for r in sub) / len(sub)
                print(f"    {intent_label}: avg={sub_avg:.3f} (n={len(sub)})")

    OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull per-row answers + scores saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
