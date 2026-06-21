"""Repair missing Ragas faithfulness scores without rerunning other metrics."""

from __future__ import annotations

import argparse
import ast
import logging
import os
import time
from pathlib import Path
from typing import Any

# Disable LangSmith/LangChain tracing before LangChain or Ragas objects initialize.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

import pandas as pd
from datasets import Dataset
from langchain_ollama import ChatOllama
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.run_config import RunConfig

from evaluate_rag_with_ragas_ollama import (
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    strip_citation_footer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_JUDGE_NUM_PREDICT = int(os.getenv("RAGAS_JUDGE_NUM_PREDICT", "4096"))
DEFAULT_REPAIR_COOLDOWN_TIME = int(os.getenv("RAGAS_REPAIR_COOLDOWN_TIME", "60"))


def setup_faithfulness_evaluator(judge_num_predict: int) -> tuple[ChatOllama, RunConfig]:
    evaluator_llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_CHAT_MODEL,
        temperature=0,
        num_thread=1,
        num_predict=judge_num_predict,
        num_ctx=8192,
    )
    run_config = RunConfig(
        timeout=600,
        max_retries=2,
        max_wait=10,
        max_workers=1,
    )
    return evaluator_llm, run_config


def parse_retrieved_contexts(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [str(value)]


def build_faithfulness_row(row: pd.Series) -> dict[str, Any]:
    return {
        "user_input": str(row.get("user_input", "")),
        "retrieved_contexts": parse_retrieved_contexts(row.get("retrieved_contexts", [])),
        "response": strip_citation_footer(str(row.get("response", ""))),
        "reference": str(row.get("reference", "")),
    }


def run_faithfulness_for_row(
    row: pd.Series,
    evaluator_llm: ChatOllama,
    run_config: RunConfig,
) -> tuple[Any, str]:
    """Return a repaired faithfulness score and a logged error string if it fails."""
    try:
        dataset = Dataset.from_pandas(
            pd.DataFrame([build_faithfulness_row(row)]),
            preserve_index=False,
        )
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness],
            llm=evaluator_llm,
            run_config=run_config,
            raise_exceptions=True,
            show_progress=True,
        )
        return result.to_pandas().iloc[0].get("faithfulness", pd.NA), ""
    except Exception as exc:
        logger.exception("Faithfulness repair failed for source_row %s", row.get("source_row"))
        return pd.NA, f"faithfulness_repair: {type(exc).__name__}: {exc}"


def write_results(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun only missing Ragas faithfulness scores with citation footers stripped."
    )
    parser.add_argument("--input", default="ragas_eval_results_v3.csv", help="Results CSV to repair")
    parser.add_argument(
        "--missing-input",
        default="missing_faithfulness_rows.csv",
        help="Optional pre-filtered missing rows CSV; falls back to filtering --input",
    )
    parser.add_argument("--output", default=None, help="Output CSV path; defaults to overwriting --input")
    parser.add_argument(
        "--cooldown",
        type=int,
        default=DEFAULT_REPAIR_COOLDOWN_TIME,
        help="Seconds to sleep after each attempted faithfulness row",
    )
    parser.add_argument(
        "--judge-num-predict",
        type=int,
        default=DEFAULT_JUDGE_NUM_PREDICT,
        help="Ollama num_predict for the Ragas judge; higher prevents truncated JSON",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    missing_path = Path(args.missing_input)

    df = pd.read_csv(input_path)
    before_missing = int(df["faithfulness"].isna().sum())

    if missing_path.exists():
        missing_df = pd.read_csv(missing_path)
    else:
        missing_df = df[df["faithfulness"].isna()].copy()

    missing_source_rows = set(df.loc[df["faithfulness"].isna(), "source_row"].tolist())
    rows_to_repair = missing_df[missing_df["source_row"].isin(missing_source_rows)].copy()

    if rows_to_repair.empty:
        print(f"Missing faithfulness before: {before_missing}")
        print(f"Missing faithfulness after: {before_missing}")
        print("No missing faithfulness rows to repair.")
        return

    evaluator_llm, run_config = setup_faithfulness_evaluator(args.judge_num_predict)

    for attempt_number, (_, row) in enumerate(rows_to_repair.iterrows(), start=1):
        source_row = row["source_row"]
        logger.info(
            "Repairing faithfulness for source_row %s (%s/%s)",
            source_row,
            attempt_number,
            len(rows_to_repair),
        )

        score, error = run_faithfulness_for_row(row, evaluator_llm, run_config)
        row_mask = df["source_row"] == source_row

        if pd.notna(score):
            df.loc[row_mask, "faithfulness"] = score
            df.loc[row_mask, "evaluation_error"] = ""
            logger.info("Repaired source_row %s faithfulness=%s", source_row, score)
        else:
            df.loc[row_mask, "evaluation_error"] = error
            logger.warning("Source_row %s still missing faithfulness: %s", source_row, error)

        write_results(df, output_path)
        after_missing = int(df["faithfulness"].isna().sum())
        print(
            f"Progress: source_row={source_row}, "
            f"missing faithfulness {before_missing} -> {after_missing}"
        )

        if attempt_number < len(rows_to_repair) and args.cooldown > 0:
            print(f"Thermal cooldown active for {args.cooldown}s...")
            time.sleep(args.cooldown)

    after_missing = int(df["faithfulness"].isna().sum())
    print(f"Missing faithfulness before: {before_missing}")
    print(f"Missing faithfulness after: {after_missing}")
    print(f"Saved repaired results to {output_path}")


if __name__ == "__main__":
    main()