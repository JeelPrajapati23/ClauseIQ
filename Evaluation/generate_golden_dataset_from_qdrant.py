"""
Generate a synthetic Ragas-ready golden dataset from a local Qdrant collection.

The script:
1. Scrolls Qdrant payloads and extracts raw chunk text.
2. Samples diverse source chunks.
3. Uses local Ollama llama3.1:8b to create question/ground_truth pairs.
4. Saves progress incrementally to golden_dataset.csv.
5. Sleeps between micro-batches to reduce thermal load on a laptop.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "pdf_knowledge_base")
PAYLOAD_TEXT_KEY = "page_content"
TOTAL_QUESTIONS_TARGET = 50
BATCH_SIZE = 2
COOLDOWN_TIME = 45

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
OUTPUT_FILE = os.getenv("GOLDEN_DATASET_OUTPUT_FILE", "golden_dataset.csv")
OVERWRITE_OUTPUT = os.getenv("OVERWRITE_GOLDEN_DATASET", "false").lower() == "true"

SCROLL_PAGE_SIZE = 256
MIN_CHUNK_CHARS = 350
MAX_CONTEXT_CHARS = 3000
MAX_GENERATION_RETRIES = 2


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a legal document auditor. Given a snippet of text from an agreement, generate a highly realistic, specific user query and its perfect matching answer based strictly on the text.

CRITICAL:
- Do not make up facts.
- Avoid generic lookup trivia questions like "What is Exhibit 10?". Instead, phrase the question around the legal substance, conditions, rights, restrictions, deadlines, payment terms, ownership, termination, confidentiality, assignment, indemnity, governing law, or obligations.
- The answer must be fully supported by the provided snippet.
- If the snippet does not contain enough substance for a high-quality legal question, return {"question": "", "ground_truth": ""}.
- Output your response strictly in valid JSON format with keys: "question" and "ground_truth".
"""


@dataclass(frozen=True)
class SourceChunk:
    text: str
    source: str
    page: str


def normalize_text(text: str) -> str:
    """Normalize whitespace and remove common extraction artifacts."""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_fingerprint(text: str) -> str:
    """Create a stable hash used for deduplication."""
    normalized = normalize_text(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_text_from_payload(payload: dict[str, Any]) -> str:
    """Extract raw text from the configured Qdrant payload key."""
    value = payload.get(PAYLOAD_TEXT_KEY)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def extract_metadata(payload: dict[str, Any]) -> tuple[str, str]:
    """Best-effort source/page extraction for logging and diversity sampling."""
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    source = (
        metadata.get("source")
        or payload.get("source")
        or metadata.get("file_name")
        or "unknown_source"
    )
    page = metadata.get("page") or payload.get("page") or "unknown_page"
    return str(source), str(page)


def fetch_qdrant_chunks() -> list[SourceChunk]:
    """Scroll all Qdrant points and collect unique text chunks from payloads."""
    client = QdrantClient(url=QDRANT_URL)
    chunks: list[SourceChunk] = []
    seen_hashes: set[str] = set()
    offset = None

    logger.info("Connecting to Qdrant at %s", QDRANT_URL)
    logger.info("Scrolling collection '%s'", COLLECTION_NAME)

    while True:
        try:
            points, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        except UnexpectedResponse as exc:
            raise RuntimeError(
                f"Qdrant rejected the scroll request for collection "
                f"'{COLLECTION_NAME}'. Confirm the collection exists."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed while scrolling Qdrant: {exc}") from exc

        for point in points:
            payload = point.payload or {}
            text = normalize_text(extract_text_from_payload(payload))
            if len(text) < MIN_CHUNK_CHARS:
                continue

            fingerprint = text_fingerprint(text)
            if fingerprint in seen_hashes:
                continue

            seen_hashes.add(fingerprint)
            source, page = extract_metadata(payload)
            chunks.append(SourceChunk(text=text, source=source, page=page))

        if offset is None:
            break

    logger.info("Collected %s unique usable chunks from Qdrant.", len(chunks))
    return chunks


def select_diverse_chunks(
    chunks: list[SourceChunk], target_count: int
) -> list[SourceChunk]:
    """
    Select chunks across different source files before taking multiple chunks
    from the same source. This avoids generating 50 near-duplicate questions
    from one long agreement.
    """
    by_source: dict[str, list[SourceChunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source, []).append(chunk)

    for source_chunks in by_source.values():
        source_chunks.sort(key=lambda chunk: (chunk.page, len(chunk.text)))

    selected: list[SourceChunk] = []
    source_names = sorted(by_source)

    while len(selected) < target_count:
        added_this_round = False
        for source in source_names:
            if by_source[source]:
                selected.append(by_source[source].pop(0))
                added_this_round = True
                if len(selected) >= target_count:
                    break

        if not added_this_round:
            break

    logger.info("Selected %s diverse chunks for generation.", len(selected))
    return selected


def setup_local_llm() -> ChatOllama:
    """Initialize local Ollama generation model."""
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.5,
        num_thread=1,
        num_predict=512,
    )


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """
    Parse a JSON object from the LLM response, tolerating markdown fences or
    minor wrapper text while still requiring valid JSON internally.
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def generate_qa_pair(llm: ChatOllama, chunk: SourceChunk) -> dict[str, str]:
    """Generate one question/ground_truth pair from a source chunk."""
    context = chunk.text[:MAX_CONTEXT_CHARS]
    human_prompt = f"""SOURCE:
{Path(chunk.source).name}, page {chunk.page}

TEXT SNIPPET:
{context}
"""

    last_error: Exception | None = None
    for attempt in range(1, MAX_GENERATION_RETRIES + 1):
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=human_prompt),
                ]
            )
            parsed = extract_json_object(str(response.content))
            question = normalize_text(str(parsed.get("question", "")))
            ground_truth = normalize_text(str(parsed.get("ground_truth", "")))

            if not question or not ground_truth:
                raise ValueError("LLM returned an empty question or ground_truth.")
            if question == ground_truth:
                raise ValueError("Question and ground_truth should not be identical.")

            return {
                "question": question,
                "ground_truth": ground_truth,
            }
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Generation attempt %s failed for source=%s page=%s: %s",
                attempt,
                Path(chunk.source).name,
                chunk.page,
                exc,
            )
            time.sleep(2)

    raise RuntimeError(
        f"Failed to generate a valid QA pair after {MAX_GENERATION_RETRIES} attempts."
    ) from last_error


def output_has_header(output_path: Path) -> bool:
    """Return True when the output CSV already exists and has content."""
    return output_path.exists() and output_path.stat().st_size > 0


def count_existing_rows(output_path: Path) -> int:
    """Count existing generated rows so reruns can resume safely."""
    if not output_has_header(output_path):
        return 0
    try:
        return len(pd.read_csv(output_path))
    except Exception:
        logger.warning("Could not read existing %s; assuming zero rows.", output_path)
        return 0


def load_existing_questions(output_path: Path) -> set[str]:
    """Load already-generated questions so reruns avoid duplicate rows."""
    if not output_has_header(output_path):
        return set()
    try:
        df = pd.read_csv(output_path, usecols=["question"])
    except Exception:
        return set()
    return {normalize_text(str(question)).lower() for question in df["question"]}


def append_generated_row(row: dict[str, str], output_path: Path) -> None:
    """Append one generated row immediately so progress is never lost."""
    df = pd.DataFrame([row], columns=["question", "ground_truth"])
    df.to_csv(
        output_path,
        mode="a",
        header=not output_has_header(output_path),
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        encoding="utf-8",
    )


def main() -> None:
    output_path = Path(OUTPUT_FILE)
    if OVERWRITE_OUTPUT and output_path.exists():
        output_path.unlink()
        logger.info("Removed existing %s because OVERWRITE_GOLDEN_DATASET=true", output_path)

    existing_count = count_existing_rows(output_path)
    existing_questions = load_existing_questions(output_path)

    if existing_count >= TOTAL_QUESTIONS_TARGET:
        logger.info(
            "%s already has %s rows, meeting target=%s. Nothing to do.",
            output_path,
            existing_count,
            TOTAL_QUESTIONS_TARGET,
        )
        return

    chunks = fetch_qdrant_chunks()
    if not chunks:
        raise RuntimeError(
            f"No usable chunks found in collection '{COLLECTION_NAME}' using "
            f"payload key '{PAYLOAD_TEXT_KEY}'."
        )

    remaining_target = TOTAL_QUESTIONS_TARGET - existing_count
    selected_chunks = select_diverse_chunks(chunks, remaining_target * 3)
    llm = setup_local_llm()

    generated_count = existing_count
    processed_in_current_batch = 0

    for chunk_index, chunk in enumerate(selected_chunks, start=1):
        if generated_count >= TOTAL_QUESTIONS_TARGET:
            break

        try:
            row = generate_qa_pair(llm, chunk)
            normalized_question = row["question"].lower()
            if normalized_question in existing_questions:
                logger.info("Skipping duplicate generated question: %s", row["question"])
                continue

            append_generated_row(row, output_path)
            existing_questions.add(normalized_question)
            generated_count += 1
            processed_in_current_batch += 1
            logger.info(
                "Generated %s/%s from %s page %s",
                generated_count,
                TOTAL_QUESTIONS_TARGET,
                Path(chunk.source).name,
                chunk.page,
            )
        except Exception as exc:
            logger.exception(
                "Skipping chunk %s from source=%s page=%s after failure: %s",
                chunk_index,
                Path(chunk.source).name,
                chunk.page,
                exc,
            )
            continue

        if processed_in_current_batch >= BATCH_SIZE:
            processed_in_current_batch = 0
            print("🛑 Thermal Cooldown active. Allowing hardware to rest...")
            time.sleep(COOLDOWN_TIME)

    logger.info(
        "Finished generation with %s/%s rows in %s",
        generated_count,
        TOTAL_QUESTIONS_TARGET,
        output_path.resolve(),
    )


if __name__ == "__main__":
    main()
