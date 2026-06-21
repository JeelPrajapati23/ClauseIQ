"""
Thermally-throttled Ragas evaluation for a local Ollama-backed RAG pipeline.

Expected input CSV:
    golden_dataset.csv with columns: question, ground_truth

Output:
    ragas_eval_results.csv, appended after every micro-batch.
"""

from __future__ import annotations

import logging
import os

# Disable external tracing before LangChain/Ragas initialize; failed trace uploads slow local evals.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""
import re
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.run_config import RunConfig

load_dotenv()



# Keep native math/thread pools quiet before the evaluation workload starts.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


BATCH_SIZE = int(os.getenv("RAGAS_BATCH_SIZE", "2"))
COOLDOWN_TIME = int(os.getenv("RAGAS_COOLDOWN_TIME", "60"))
OUTPUT_FILE = os.getenv("RAGAS_OUTPUT_FILE", "ragas_eval_results.csv")
INPUT_FILE = os.getenv("RAGAS_INPUT_FILE", "golden_dataset.csv")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
JUDGE_NUM_PREDICT = int(os.getenv("RAGAS_JUDGE_NUM_PREDICT", "4096"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

_CITATION_FOOTER_RE = re.compile(r"(?:\s*\[\d+\])*\s*Sources Used:.*\Z", re.DOTALL)


def strip_citation_footer(text: str) -> str:
    """
    Remove generated citation metadata before Ragas faithfulness atomizes claims.

    The original response remains useful for display and other metrics; this cleaned
    version is only for faithfulness, where the footer can be misread as a claim.
    """
    if not isinstance(text, str):
        return ""
    return _CITATION_FOOTER_RE.sub("", text).rstrip()


def setup_local_evaluator() -> tuple[ChatOllama, Any, RunConfig]:
    """Create the evaluator LLM and reuse your pipeline embeddings for Ragas."""
    from app.database import embeddings as pipeline_embeddings
    evaluator_llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_CHAT_MODEL,
        temperature=0,
        num_thread=1,
        num_predict=JUDGE_NUM_PREDICT,
        num_ctx=8192,
    )

    run_config = RunConfig(
        timeout=600,
        max_retries=2,
        max_wait=10,
        max_workers=1,
    )

    return evaluator_llm, pipeline_embeddings, run_config


@lru_cache(maxsize=1)
def get_pipeline_retriever():
    """Build your production hybrid retriever once per evaluation run."""
    from app.database import get_reranking_retriever

    return get_reranking_retriever()


def run_my_rag_pipeline(question: str) -> dict[str, Any]:
    """Run the same retriever and answer generator used by the FastAPI app."""
    from app.generator import generate_answer

    retriever = get_pipeline_retriever()
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return {
            "answer": (
                "I cannot answer this based on the provided documents. "
                "No relevant context was found."
            ),
            "contexts": [],
        }

    contexts = [doc.page_content for doc in retrieved_docs]
    context_parts = []

    for doc in retrieved_docs:
        source_file = Path(str(doc.metadata.get("source", "Unknown"))).name
        page = doc.metadata.get("page", "Unknown")
        context_parts.append(
            f"Source: {source_file}, Page: {page}\nContent: {doc.page_content}"
        )

    formatted_context = "\n---\n".join(context_parts)
    answer = generate_answer(question, formatted_context)

    return {
        "answer": answer,
        "contexts": contexts,
    }


def validate_input(df: pd.DataFrame) -> None:
    """Fail fast if the golden dataset is missing required columns."""
    required_columns = {"question", "ground_truth"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"{INPUT_FILE} is missing required columns: {sorted(missing_columns)}"
        )


def build_ragas_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Run the RAG pipeline for a micro-batch and map rows to Ragas columns."""
    records: list[dict[str, Any]] = []

    for row_number, row in batch_df.iterrows():
        question = str(row["question"])
        ground_truth = str(row["ground_truth"])

        try:
            rag_output = run_my_rag_pipeline(question)
            answer = str(rag_output.get("answer", ""))
            contexts = rag_output.get("contexts", [])

            if not isinstance(contexts, list) or not all(
                isinstance(context, str) for context in contexts
            ):
                raise TypeError("RAG pipeline must return contexts as list[str].")

            pipeline_error = ""
        except Exception as exc:
            logger.exception("RAG pipeline failed for row %s", row_number)
            answer = ""
            contexts = []
            pipeline_error = f"{type(exc).__name__}: {exc}"

        records.append(
            {
                "source_row": row_number,
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": ground_truth,
                "pipeline_error": pipeline_error,
            }
        )

    return pd.DataFrame.from_records(records)


def append_results(results_df: pd.DataFrame, output_path: Path) -> None:
    """Append one batch of results to disk, writing headers only once."""
    write_header = not output_path.exists()
    results_df.to_csv(
        output_path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8",
    )


def make_failure_results(batch_df: pd.DataFrame, error: Exception) -> pd.DataFrame:
    """Persist failed evaluation rows so the run can continue after bad parses."""
    failed_df = batch_df.copy()
    failed_df["faithfulness"] = pd.NA
    failed_df["answer_relevancy"] = pd.NA
    failed_df["context_precision"] = pd.NA
    failed_df["evaluation_error"] = f"{type(error).__name__}: {error}"
    failed_df["evaluation_traceback"] = traceback.format_exc(limit=8)
    return failed_df


def is_guardrail_refusal(answer: str) -> bool:
    """Detect safe fallback answers that should not be penalized as hallucinations."""
    normalized = answer.lower()
    refusal_markers = (
        "cannot answer this based on the provided documents",
        "provided context chunks do not contain",
        "therefore an answer cannot be generated based on these documents",
    )
    return any(marker in normalized for marker in refusal_markers)


def make_manual_refusal_results(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Assign faithful-refusal rows a manual faithfulness score."""
    manual_df = batch_df.copy()
    manual_df["faithfulness"] = 1.0
    manual_df["answer_relevancy"] = pd.NA
    manual_df["context_precision"] = pd.NA
    manual_df["evaluation_error"] = (
        "manual_refusal_scoring: guardrail refusal is faithful by policy; "
        "answer_relevancy/context_precision skipped for fallback state"
    )
    return manual_df


def evaluate_metric_group(
    row_df: pd.DataFrame,
    metrics: list[Any],
    evaluator_llm: ChatOllama,
    evaluator_embeddings: Any,
    run_config: RunConfig,
    *,
    strip_response_for_faithfulness: bool = False,
) -> dict[str, Any]:
    """Evaluate one row for a metric group and let callers record real failures."""
    ragas_df = row_df.drop(columns=["pipeline_error"], errors="ignore").copy()
    if strip_response_for_faithfulness:
        ragas_df["response"] = ragas_df["response"].map(strip_citation_footer)

    dataset = Dataset.from_pandas(ragas_df, preserve_index=False)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
        raise_exceptions=True,
        show_progress=True,
    )
    return result.to_pandas().iloc[0].to_dict()


def evaluate_ragas_row(
    row: pd.Series,
    evaluator_llm: ChatOllama,
    evaluator_embeddings: Any,
    run_config: RunConfig,
) -> dict[str, Any]:
    """Evaluate one non-refusal row while preserving partial metric failures."""
    row_df = pd.DataFrame([row.to_dict()])
    output = row.to_dict()
    output["faithfulness"] = pd.NA
    output["answer_relevancy"] = pd.NA
    output["context_precision"] = pd.NA
    errors: list[str] = []

    try:
        faithfulness_scores = evaluate_metric_group(
            row_df,
            [faithfulness],
            evaluator_llm,
            evaluator_embeddings,
            run_config,
            strip_response_for_faithfulness=True,
        )
        output["faithfulness"] = faithfulness_scores.get("faithfulness", pd.NA)
    except Exception as exc:
        logger.exception(
            "Ragas faithfulness failed for source_row %s", row.get("source_row")
        )
        errors.append(f"faithfulness: {type(exc).__name__}: {exc}")

    try:
        other_scores = evaluate_metric_group(
            row_df,
            [answer_relevancy, context_precision],
            evaluator_llm,
            evaluator_embeddings,
            run_config,
        )
        output["answer_relevancy"] = other_scores.get("answer_relevancy", pd.NA)
        output["context_precision"] = other_scores.get("context_precision", pd.NA)
    except Exception as exc:
        logger.exception(
            "Ragas answer/context metrics failed for source_row %s",
            row.get("source_row"),
        )
        errors.append(f"answer_context: {type(exc).__name__}: {exc}")

    output["evaluation_error"] = " | ".join(errors)
    return output


def evaluate_batch(
    batch_df: pd.DataFrame,
    evaluator_llm: ChatOllama,
    evaluator_embeddings: Any,
    run_config: RunConfig,
) -> pd.DataFrame:
    """Evaluate one micro-batch with Ragas."""
    refusal_mask = batch_df["response"].map(is_guardrail_refusal)
    manual_results = make_manual_refusal_results(batch_df[refusal_mask])
    rows_for_ragas = batch_df[~refusal_mask]

    if rows_for_ragas.empty:
        return manual_results

    results_df = pd.DataFrame.from_records(
        [
            evaluate_ragas_row(row, evaluator_llm, evaluator_embeddings, run_config)
            for _, row in rows_for_ragas.iterrows()
        ]
    )
    return pd.concat([manual_results, results_df], ignore_index=True)


def main() -> None:
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find {input_path.resolve()}. Create it with question and "
            "ground_truth columns before running this script."
        )

    golden_df = pd.read_csv(input_path)
    validate_input(golden_df)

    evaluator_llm, evaluator_embeddings, run_config = setup_local_evaluator()
    total_rows = len(golden_df)

    logger.info(
        "Starting Ragas evaluation: %s rows, batch_size=%s, cooldown=%ss, output=%s",
        total_rows,
        BATCH_SIZE,
        COOLDOWN_TIME,
        output_path,
    )

    for batch_start in range(0, total_rows, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_rows)
        batch_number = (batch_start // BATCH_SIZE) + 1
        batch_df = golden_df.iloc[batch_start:batch_end].copy()

        logger.info(
            "Processing batch %s covering rows %s-%s",
            batch_number,
            batch_start,
            batch_end - 1,
        )

        prepared_batch = build_ragas_batch(batch_df)

        try:
            results_df = evaluate_batch(
                prepared_batch,
                evaluator_llm,
                evaluator_embeddings,
                run_config,
            )
        except Exception as exc:
            logger.exception("Ragas evaluation failed for batch %s", batch_number)
            results_df = make_failure_results(prepared_batch, exc)

        append_results(results_df, output_path)

        logger.info(
            "Batch %s complete. Saved %s rows to %s.",
            batch_number,
            len(results_df),
            output_path,
        )

        print("ðŸ›‘ Thermal Cooldown active. Allowing hardware to rest...")
        time.sleep(COOLDOWN_TIME)

    logger.info("Evaluation complete. Results saved to %s", output_path.resolve())


if __name__ == "__main__":
    main()


