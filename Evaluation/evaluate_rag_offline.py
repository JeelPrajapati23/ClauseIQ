"""
Ragas evaluation of the production RAG pipeline (app.database.get_reranking_retriever
+ app.generator.stream_answer), judged by a Groq-hosted model — deliberately a
different model than the pipeline's own generation model, so the judge isn't grading
its own generations.

Retrieval still runs against the real pipeline, which means Qdrant must be
running and populated (see ingest_eval_corpus.py) and the pipeline's own
GROQ_API_KEY / COHERE_API_KEY must be set — the Ragas judge reuses GROQ_API_KEY
by default, or set RAGAS_GROQ_API_KEYS (comma-separated) to give the judge its
own pool of keys to rotate through on rate-limit errors (same model throughout,
see RotatingChatGroq below).

Expected input JSON (see Evaluation/golden_qa_set.json):
    list of {"question", "ground_truth", "reference_contexts"?, "intent"?,
             "synthesizer"?, "doc_category"?}

Output:
    Evaluation/ragas_eval_results_offline.csv, appended after every
    micro-batch. Re-running the script skips source rows already present
    in that file, so an interrupted run can be resumed for free.

Generation backend: defaults to the production pipeline (Groq/stream_answer). Set
RAGAS_GENERATION_BACKEND=gemini to swap only the answer-generation call to a Gemini
candidate model (retrieval, verifier, and the Ragas judge are all unchanged) — for a
full Ragas run on a candidate model that a fast proxy comparison alone can't give you
(no answer_relevancy/context_precision/context_recall there). Requires GEMINI_API_KEY
(comma-separated keys supported; the script's resumability means a run can also span
multiple days if needed). Optional RAGAS_GEMINI_MODEL. Needs `pip install google-genai`
(see gemini_rotation.py). Point RAGAS_OUTPUT_FILE at a new file so this doesn't
overwrite the production-model baseline.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys

# Disable external tracing before LangChain/Ragas initialize; failed trace uploads slow local evals.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from groq import RateLimitError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_groq import ChatGroq
from pydantic import PrivateAttr

try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    # ragas (as of 0.4.x) unconditionally imports ChatVertexAI at module load time
    # (ragas/llms/base.py), but langchain-community has since dropped that submodule
    # in favor of the standalone langchain-google-vertexai package. We never use
    # VertexAI as a judge here (see RotatingChatGroq below), so stub just enough to
    # satisfy the import rather than pinning to an older, less compatible
    # langchain-community across the whole environment.
    import types

    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class _VertexAIUnavailable:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "ChatVertexAI is unavailable (langchain-community dropped this "
                "integration); this eval harness only uses Groq judges."
            )

    _vertexai_stub.ChatVertexAI = _VertexAIUnavailable
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# Keep native math/thread pools quiet before the evaluation workload starts.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


BATCH_SIZE = int(os.getenv("RAGAS_BATCH_SIZE", "2"))
COOLDOWN_TIME = int(os.getenv("RAGAS_COOLDOWN_TIME", "10"))
OUTPUT_FILE = os.getenv(
    "RAGAS_OUTPUT_FILE", str(Path(__file__).resolve().parent / "ragas_eval_results_scoped.csv")
)
INPUT_FILE = os.getenv(
    "RAGAS_INPUT_FILE", str(Path(__file__).resolve().parent / "golden_qa_set.json")
)
GOLDEN_DOC_MAP_FILE = os.getenv(
    "RAGAS_GOLDEN_DOC_MAP_FILE", str(Path(__file__).resolve().parent / "golden_doc_map.json")
)

EVAL_COLLECTION_NAME = os.getenv("EVAL_COLLECTION_NAME", "eval_knowledge_base")
EVAL_USER_ID = os.getenv("EVAL_USER_ID", "eval")
# Mirrors app/main.py's MAX_FULL_DOC_CHARS.
MAX_FULL_DOC_CHARS = 20000

# Deliberately defaults to a different model than the pipeline's own generation
# model, so the judge isn't grading its own generations.
GROQ_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-120b")
JUDGE_NUM_PREDICT = int(os.getenv("RAGAS_JUDGE_NUM_PREDICT", "4096"))

# Which model answers the questions — independent of GROQ_JUDGE_MODEL above, which always
# grades the answer regardless of what produced it. "groq" (default) uses the real
# production path (app.generator.stream_answer); "gemini" swaps in a Gemini candidate via
# the rotating multi-key client (see gemini_rotation.py and the module docstring above).
RAGAS_GENERATION_BACKEND = os.getenv("RAGAS_GENERATION_BACKEND", "groq").strip().lower()
RAGAS_GEMINI_MODEL = os.getenv("RAGAS_GEMINI_MODEL", "gemini-2.5-flash")
_gemini_client = None


def _get_gemini_client():
    """Lazy singleton so a normal (groq) eval run never needs GEMINI_API_KEY or the
    google-genai package installed."""
    global _gemini_client
    if _gemini_client is None:
        from gemini_rotation import RotatingGeminiClient, load_gemini_keys

        keys = load_gemini_keys()
        if not keys:
            raise RuntimeError(
                "RAGAS_GENERATION_BACKEND=gemini requires GEMINI_API_KEY "
                "(comma-separated for multiple keys)."
            )
        _gemini_client = RotatingGeminiClient(RAGAS_GEMINI_MODEL, keys)
        logger.info(
            "Gemini generation backend active: %s key(s), model=%s.",
            len(keys), RAGAS_GEMINI_MODEL,
        )
    return _gemini_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _load_judge_api_keys() -> list[str]:
    """Parse RAGAS_GROQ_API_KEYS (comma-separated) for judge key rotation, falling back
    to the single GROQ_API_KEY the rest of the pipeline already uses.
    """
    raw = os.getenv("RAGAS_GROQ_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if keys:
        return keys
    single_key = os.getenv("GROQ_API_KEY")
    if not single_key:
        raise RuntimeError("Set GROQ_API_KEY or RAGAS_GROQ_API_KEYS before running the eval.")
    return [single_key]


class RotatingChatGroq(BaseChatModel):
    """Same judge model, multiple Groq API keys: on a rate-limit error, rotates to the
    next key and retries instead of stalling the whole eval run on one exhausted key.
    """

    _backends: list[ChatGroq] = PrivateAttr()
    _index: int = PrivateAttr(default=0)

    def __init__(self, model: str, api_keys: list[str], **kwargs: Any) -> None:
        super().__init__()
        self._backends = [ChatGroq(model=model, api_key=key, **kwargs) for key in api_keys]
        self._index = 0

    @property
    def _llm_type(self) -> str:
        return "rotating-chat-groq"

    def _rotate(self, reason: str) -> None:
        next_index = (self._index + 1) % len(self._backends)
        logger.warning(
            "Groq judge key #%s rate-limited (%s); rotating to key #%s.",
            self._index, reason, next_index,
        )
        self._index = next_index

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        last_exc: Exception | None = None
        for _ in range(len(self._backends)):
            try:
                ai_message = self._backends[self._index].invoke(messages, stop=stop, **kwargs)
                return ChatResult(generations=[ChatGeneration(message=ai_message)])
            except RateLimitError as exc:
                last_exc = exc
                self._rotate(str(exc))
        raise last_exc

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        last_exc: Exception | None = None
        for _ in range(len(self._backends)):
            try:
                ai_message = await self._backends[self._index].ainvoke(messages, stop=stop, **kwargs)
                return ChatResult(generations=[ChatGeneration(message=ai_message)])
            except RateLimitError as exc:
                last_exc = exc
                self._rotate(str(exc))
        raise last_exc


def setup_evaluator() -> tuple[BaseChatModel, Any, RunConfig]:
    """Create the Groq-hosted evaluator LLM and reuse the pipeline's own embeddings for Ragas."""
    from app.database import embeddings as pipeline_embeddings

    api_keys = _load_judge_api_keys()
    evaluator_llm = RotatingChatGroq(
        model=GROQ_JUDGE_MODEL,
        api_keys=api_keys,
        temperature=0,
        max_tokens=JUDGE_NUM_PREDICT,
        # gpt-oss-120b spends part of max_tokens on hidden reasoning before writing
        # output — same issue that hit app/generator.py's gpt-oss-20b calls (see
        # that file's reasoning_effort="low"). Left unset here, it caused
        # LLMDidNotFinishException/413s on ANALYTICAL rows' wider-context judge
        # calls during the first real eval run.
        reasoning_effort="low",
    )
    if len(api_keys) > 1:
        logger.info("Judge key rotation active: %s Groq API keys available.", len(api_keys))

    run_config = RunConfig(
        timeout=600,
        max_retries=2,
        max_wait=10,
        max_workers=1,
    )

    return evaluator_llm, pipeline_embeddings, run_config


def _format_page_range(pages: list) -> str:
    """Merge consecutive page numbers into compact ranges: [3,4,5,7] -> 'pp. 3-5, 7'. Mirrors app.main."""
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


@lru_cache(maxsize=None)
def get_pipeline_retriever(intent_value: str, document_filter: tuple = ()):
    """Build one production hybrid retriever per (intent, document_filter), matching
    app.main's k values, cached so repeated filters across rows are built only once.
    """
    from app.database import get_reranking_retriever
    from app.generator import QueryIntent

    is_analytical = intent_value == QueryIntent.ANALYTICAL.value
    return get_reranking_retriever(
        user_id=EVAL_USER_ID,
        collection_name=EVAL_COLLECTION_NAME,
        # Mirrors app/main.py's /ask/ route: initial_k=20 for both intents (raised FACT
        # from 10), final_k=6 for FACT (raised from 3) — see that file's comments for
        # the sweep results.
        initial_k=20,
        final_k=5 if is_analytical else 6,
        document_filter=list(document_filter) or None,
    )


def run_my_rag_pipeline(question: str, document_filter: tuple = ()) -> dict[str, Any]:
    """Run the same retriever + streaming generator used by the FastAPI /ask/ route.

    document_filter scopes retrieval to specific source_file(s) — used when the golden
    row's originating contract is known (see golden_doc_map.json), since CUAD-style
    questions mean "in this contract," not "search the whole eval corpus."
    """
    from app.database import get_full_document_context
    from app.generator import (
        classify_intent,
        load_prompt,
        needs_full_document_read,
        stream_answer,
        QueryIntent,
        _ANALYTICAL_PROMPT_FILE,
    )

    intent = classify_intent(question)
    # Mirrors main.py's /ask/ route: clause-category questions scoped to exactly one
    # document bypass retrieval in favor of reading the whole document (see
    # generator.needs_full_document_read's docstring), capped at MAX_FULL_DOC_CHARS.
    full_doc_docs = (
        get_full_document_context(user_id=EVAL_USER_ID, source_file=document_filter[0], collection_name=EVAL_COLLECTION_NAME)
        if needs_full_document_read(question) and len(document_filter) == 1
        else None
    )
    if full_doc_docs and sum(len(d.page_content) for d in full_doc_docs) <= MAX_FULL_DOC_CHARS:
        retrieved_docs = full_doc_docs
    else:
        retriever = get_pipeline_retriever(intent.value, document_filter)
        retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return {
            "answer": (
                "I cannot answer this based on the provided documents. "
                "No relevant context was found."
            ),
            "contexts": [],
        }

    # retriever.invoke() already returns parent-context documents (see
    # ThresholdReranker in app.database), so page_content here is exactly
    # what the LLM is shown — no extra parent-swap needed.
    contexts = [doc.page_content for doc in retrieved_docs]
    context_parts = []
    for doc in retrieved_docs:
        sf = doc.metadata.get("source_file", Path(str(doc.metadata.get("source", "Unknown"))).name)
        all_pages = doc.metadata.get("all_pages") or [doc.metadata.get("page", "?")]
        section = doc.metadata.get("section", "")
        header = f"Source: {sf}, Pages: {_format_page_range(all_pages)}"
        if section:
            header += f", Section: {section}"
        context_parts.append(f"{header}\nContent: {doc.page_content}\n")
    formatted_context = "\n---\n".join(context_parts)

    if RAGAS_GENERATION_BACKEND == "gemini":
        # Same prompt selection stream_answer applies internally — only the model
        # generating the answer changes; retrieval and prompt content stay identical.
        prompt_file = _ANALYTICAL_PROMPT_FILE if intent == QueryIntent.ANALYTICAL else None
        system_prompt = load_prompt(prompt_file).replace("{context}", formatted_context)
        answer = _get_gemini_client().generate(system_prompt, question)
    else:
        answer = "".join(stream_answer(question, formatted_context, history=[], intent=intent))

    return {"answer": answer, "contexts": contexts}


def load_golden_dataset(input_path: Path) -> pd.DataFrame:
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame.from_records(records)
    df.index.name = "source_row"
    df = df.reset_index()
    df["doc_source_file"] = df["source_row"].map(load_golden_doc_map())
    return df


def load_golden_doc_map() -> dict[int, str]:
    """Map source_row -> matched contract filename, built by build_golden_doc_map.py.

    Rows with no confident match (hand-written analytical questions that paraphrase
    the source text, or guardrail rows with no source contract) fall back to
    unscoped, full-corpus retrieval — same behavior as if no map existed.
    """
    path = Path(GOLDEN_DOC_MAP_FILE)
    if not path.exists():
        logger.warning(
            "%s not found; retrieval will search the full eval corpus for every "
            "question. Run build_golden_doc_map.py first to scope per-question "
            "retrieval to the golden set's originating contract.",
            path,
        )
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v["source_file"] for k, v in raw.items() if v.get("source_file")}


def validate_input(df: pd.DataFrame) -> None:
    """Fail fast if the golden dataset is missing required columns."""
    required_columns = {"question", "ground_truth"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"{INPUT_FILE} is missing required columns: {sorted(missing_columns)}"
        )


def load_processed_rows(output_path: Path) -> set:
    """Read source_row values already scored in a prior run, so it can be resumed for free."""
    if not output_path.exists():
        return set()
    try:
        existing = pd.read_csv(output_path, usecols=["source_row"])
        return set(existing["source_row"].tolist())
    except Exception:
        logger.warning("Could not read existing %s to resume; starting from scratch.", output_path)
        return set()


def build_ragas_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Run the RAG pipeline for a micro-batch and map rows to Ragas columns."""
    records: list[dict[str, Any]] = []

    for _, row in batch_df.iterrows():
        question = str(row["question"])
        ground_truth = str(row["ground_truth"])
        doc_source_file = row.get("doc_source_file")
        document_filter = (doc_source_file,) if isinstance(doc_source_file, str) and doc_source_file else ()

        try:
            rag_output = run_my_rag_pipeline(question, document_filter)
            answer = str(rag_output.get("answer", ""))
            contexts = rag_output.get("contexts", [])

            if not isinstance(contexts, list) or not all(
                isinstance(context, str) for context in contexts
            ):
                raise TypeError("RAG pipeline must return contexts as list[str].")

            pipeline_error = ""
        except Exception as exc:
            logger.exception("RAG pipeline failed for source_row %s", row["source_row"])
            answer = ""
            contexts = []
            pipeline_error = f"{type(exc).__name__}: {exc}"

        records.append(
            {
                "source_row": row["source_row"],
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": ground_truth,
                "golden_intent": row.get("intent", ""),
                "synthesizer": row.get("synthesizer", ""),
                "doc_category": row.get("doc_category", ""),
                "doc_source_file": doc_source_file or "",
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
    failed_df["context_recall"] = pd.NA
    failed_df["evaluation_error"] = f"{type(error).__name__}: {error}"
    failed_df["evaluation_traceback"] = traceback.format_exc(limit=8)
    return failed_df


_REFUSAL_VERB_RE = re.compile(
    r"does not (mention|specify|contain|address|include|discuss|indicate|provide)"
)
# A genuine refusal is one short, direct sentence. A substantive answer can legitimately
# contain the same phrase as an aside deep in a long, structured analysis (e.g. "the
# clause is broad and does not specify particular services..." inside a multi-paragraph
# breakdown) — gating on position + overall length is what tells these apart; an
# ungated regex was verified to false-positive on exactly that kind of real answer
# before this guard was added.
_REFUSAL_VERB_MAX_POS = 150
_REFUSAL_VERB_MAX_LEN = 250


def is_guardrail_refusal(answer: str) -> bool:
    """Detect safe fallback answers (empty retrieval or LLM refusal) that shouldn't be penalized.

    Synced with main.py's _REFUSAL_MARKERS (small/fast Groq models often paraphrase the
    system prompt's exact refusal string rather than reproducing it verbatim). The exact
    phrase list is inherently incomplete — re-running the identical source_row 64 twice
    produced two different paraphrases ("does not mention any OSHA..." then "does not
    specify any OSHA...") that each individually evaded a literal-phrase list, so this
    also matches the general "does not <verb>" refusal family via regex — but only when
    it appears early in a short answer, so it can't misfire on the same phrase used as
    an aside inside a long, substantive one.
    """
    normalized = answer.lower()
    refusal_markers = (
        "cannot answer this based on the provided documents",
        "therefore an answer cannot be generated",
        "no information relevant",
        "does not contain information",
        "cannot answer this based on",
    )
    if any(marker in normalized for marker in refusal_markers):
        return True
    if len(normalized) > _REFUSAL_VERB_MAX_LEN:
        return False
    match = _REFUSAL_VERB_RE.search(normalized)
    return bool(match and match.start() <= _REFUSAL_VERB_MAX_POS)


def make_manual_refusal_results(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Assign faithful-refusal rows a manual faithfulness score."""
    manual_df = batch_df.copy()
    manual_df["faithfulness"] = 1.0
    manual_df["answer_relevancy"] = pd.NA
    manual_df["context_precision"] = pd.NA
    manual_df["context_recall"] = pd.NA
    manual_df["evaluation_error"] = (
        "manual_refusal_scoring: guardrail refusal is faithful by policy; "
        "answer_relevancy/context_precision/context_recall skipped for fallback state"
    )
    return manual_df


def evaluate_metric_group(
    row_df: pd.DataFrame,
    metrics: list[Any],
    evaluator_llm: BaseChatModel,
    evaluator_embeddings: Any,
    run_config: RunConfig,
) -> dict[str, Any]:
    """Evaluate one row for a metric group and let callers record real failures."""
    ragas_df = row_df.drop(columns=["pipeline_error"], errors="ignore").copy()
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
    evaluator_llm: BaseChatModel,
    evaluator_embeddings: Any,
    run_config: RunConfig,
) -> dict[str, Any]:
    """Evaluate one non-refusal row while preserving partial metric failures."""
    row_df = pd.DataFrame([row.to_dict()])
    output = row.to_dict()
    output["faithfulness"] = pd.NA
    output["answer_relevancy"] = pd.NA
    output["context_precision"] = pd.NA
    output["context_recall"] = pd.NA
    errors: list[str] = []

    try:
        faithfulness_scores = evaluate_metric_group(
            row_df, [faithfulness], evaluator_llm, evaluator_embeddings, run_config,
        )
        output["faithfulness"] = faithfulness_scores.get("faithfulness", pd.NA)
    except Exception as exc:
        logger.exception("Ragas faithfulness failed for source_row %s", row.get("source_row"))
        errors.append(f"faithfulness: {type(exc).__name__}: {exc}")

    try:
        # context_recall needs the same columns (user_input/retrieved_contexts/
        # reference) as context_precision, so it rides in the same batched call
        # rather than a third separate group.
        other_scores = evaluate_metric_group(
            row_df, [answer_relevancy, context_precision, context_recall],
            evaluator_llm, evaluator_embeddings, run_config,
        )
        output["answer_relevancy"] = other_scores.get("answer_relevancy", pd.NA)
        output["context_precision"] = other_scores.get("context_precision", pd.NA)
        output["context_recall"] = other_scores.get("context_recall", pd.NA)
    except Exception as exc:
        logger.exception(
            "Ragas answer/context metrics failed for source_row %s", row.get("source_row"),
        )
        errors.append(f"answer_context: {type(exc).__name__}: {exc}")

    output["evaluation_error"] = " | ".join(errors)
    return output


def evaluate_batch(
    batch_df: pd.DataFrame,
    evaluator_llm: BaseChatModel,
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
            f"Could not find {input_path.resolve()}. Run build_golden_set.py / "
            "generate_golden_dataset_v2.py first, or point RAGAS_INPUT_FILE at the right file."
        )

    golden_df = load_golden_dataset(input_path)
    validate_input(golden_df)

    already_processed = load_processed_rows(output_path)
    if already_processed:
        golden_df = golden_df[~golden_df["source_row"].isin(already_processed)]
        logger.info(
            "Resuming: %s rows already scored in %s, %s remaining.",
            len(already_processed), output_path, len(golden_df),
        )

    if golden_df.empty:
        logger.info("Nothing left to evaluate. All rows already scored in %s.", output_path)
        return

    evaluator_llm, evaluator_embeddings, run_config = setup_evaluator()
    total_rows = len(golden_df)

    logger.info(
        "Starting Ragas evaluation: %s rows, batch_size=%s, cooldown=%ss, "
        "judge=%s (Groq), generation_backend=%s, output=%s",
        total_rows, BATCH_SIZE, COOLDOWN_TIME, GROQ_JUDGE_MODEL, RAGAS_GENERATION_BACKEND, output_path,
    )
    if RAGAS_GENERATION_BACKEND == "gemini":
        _get_gemini_client()  # fail fast on a missing GEMINI_API_KEY before any retrieval work

    row_positions = list(golden_df.index)
    for batch_start in range(0, total_rows, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_rows)
        batch_number = (batch_start // BATCH_SIZE) + 1
        batch_df = golden_df.loc[row_positions[batch_start:batch_end]].copy()

        logger.info(
            "Processing batch %s covering source_rows %s",
            batch_number, batch_df["source_row"].tolist(),
        )

        prepared_batch = build_ragas_batch(batch_df)

        try:
            results_df = evaluate_batch(prepared_batch, evaluator_llm, evaluator_embeddings, run_config)
        except Exception as exc:
            logger.exception("Ragas evaluation failed for batch %s", batch_number)
            results_df = make_failure_results(prepared_batch, exc)

        append_results(results_df, output_path)

        logger.info(
            "Batch %s complete. Saved %s rows to %s.",
            batch_number, len(results_df), output_path,
        )

        is_last_batch = batch_end >= total_rows
        if not is_last_batch:
            logger.info("Cooldown active: sleeping %ss before the next batch.", COOLDOWN_TIME)
            time.sleep(COOLDOWN_TIME)

    logger.info("Evaluation complete. Results saved to %s", output_path.resolve())


if __name__ == "__main__":
    main()
