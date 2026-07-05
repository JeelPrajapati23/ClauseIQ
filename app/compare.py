import os
import json
import logging
from typing import Dict, List, Generator
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.database import embeddings, QDRANT_URL, QDRANT_API_KEY, swap_to_parent_context
from app.generator import llm

logger = logging.getLogger(__name__)

COLLECTION = "pdf_knowledge_base"
CHUNKS_PER_DOC = 5
TOKEN_LIMIT = 4000  # chars / 4 ≈ tokens; above this, summarize each doc first


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_per_doc(
    query: str,
    doc_ids: List[str],
    user_id: str,
) -> Dict[str, List[Document]]:
    """MMR retrieval per document — diverse chunks, scoped to user_id + source_file."""
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        check_compatibility=False,
    )
    results: Dict[str, List[Document]] = {}
    for doc_id in doc_ids:
        doc_filter = Filter(must=[
            FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="metadata.source_file", match=MatchValue(value=doc_id)),
        ])
        child_chunks = qdrant.max_marginal_relevance_search(
            query=query,
            k=CHUNKS_PER_DOC,
            fetch_k=CHUNKS_PER_DOC * 4,
            filter=doc_filter,
        )
        results[doc_id] = swap_to_parent_context(child_chunks)
    return results


# ── Verification ──────────────────────────────────────────────────────────────

def verify_coverage(per_doc: Dict[str, List[Document]]) -> Dict[str, bool]:
    """Returns {doc_id: has_evidence}. False means zero chunks were retrieved."""
    return {doc_id: len(chunks) > 0 for doc_id, chunks in per_doc.items()}


# ── Context management ────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _format_chunk(chunk: Document, doc_id: str) -> str:
    meta = chunk.metadata
    source = os.path.basename(meta.get("source", doc_id))
    page = meta.get("page", "?")
    section = meta.get("section", "")
    header = f"[{source} | Page {page}"
    if section:
        header += f" | {section}"
    header += "]"
    return f"{header}\n{chunk.page_content}"


def _summarize_chunks(doc_id: str, chunks: List[Document]) -> str:
    formatted = "\n\n".join(_format_chunk(c, doc_id) for c in chunks)
    msg = HumanMessage(content=(
        f"Summarise excerpts from '{doc_id}' in ≤200 words. "
        f"Preserve exact clauses, numbers, dates, and legal terms. "
        f"Include page/section references.\n\n{formatted}"
    ))
    return llm.invoke([msg]).content.strip()


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_comparison_prompt(
    query: str,
    per_doc: Dict[str, List[Document]],
    coverage: Dict[str, bool],
) -> str:
    sections: List[tuple] = []
    for doc_id, chunks in per_doc.items():
        text = "\n\n".join(_format_chunk(c, doc_id) for c in chunks) if chunks \
               else f"[No relevant content found in {doc_id}]"
        sections.append((doc_id, text))

    total_tokens = _estimate_tokens("".join(t for _, t in sections))
    if total_tokens > TOKEN_LIMIT:
        sections = []
        for doc_id, chunks in per_doc.items():
            text = _summarize_chunks(doc_id, chunks) if chunks \
                   else f"[No relevant content found in {doc_id}]"
            sections.append((doc_id, text))

    doc_blocks = ""
    for i, (doc_id, text) in enumerate(sections, 1):
        sep = "=" * 60
        doc_blocks += f"\n\n{sep}\nDOCUMENT {i}: {doc_id}\n{sep}\n{text}"

    no_evidence = [d for d, ok in coverage.items() if not ok]
    coverage_note = (
        f"\n\nNOTE: These documents returned NO relevant chunks and must be "
        f"explicitly marked 'Not found': {', '.join(no_evidence)}"
    ) if no_evidence else ""

    return (
       "You are a meticulous legal document analyst.\n"
        f"Query: {query}\n"
        f"{coverage_note}"
        f"{doc_blocks}\n\n"

        "RULES:\n"
        "1. Answer ONLY the user's question. Do not discuss legal provisions unrelated to the query. "
        "Ignore confidentiality, indemnification, insurance, governing law, arbitration, payment, and "
        "liability unless the query specifically asks about them.\n"
        "2. Ground every claim in the provided excerpts. Every factual statement must cite the document name and page.\n"
        "3. If the retrieved text does not explicitly answer the query for a document, write 'Not found in [document name]'. Never infer, speculate, or rely on common contract language.\n"
        "4. Do not mix information across documents. Evaluate each document independently before comparing them.\n"
        "5. Avoid repetition. Do not restate the same finding across multiple sections.\n"
        "6. Keep the response concise.\n\n"

        "Respond using EXACTLY these sections:\n\n"

        "## COMPARISON\n"
        "For each document:\n"
        "- State whether the queried provision exists.\n"
        "- Quote or closely paraphrase ONLY the relevant clause.\n"
        "- Cite the document name and page.\n"
        "- Maximum two bullet points per document.\n\n"

        "## COMPARISON TABLE (optional)\n"
        "Include this section ONLY when a side-by-side table meaningfully clarifies the comparison. "
        "Omit it if the differences are already clear from the COMPARISON section.\n\n"
        "When included:\n"
        "1. First identify the main legal concept in the query.\n"
        "2. Create columns that capture ONLY the sub-attributes of that concept. Examples:\n"
        "   - Query: Intellectual Property  → columns: Document | Ownership | Trademark License\n"
        "   - Query: Confidentiality        → columns: Document | Confidentiality | Survival\n"
        "   - Query: Termination            → columns: Document | Breach | Insolvency | Convenience\n"
        "   - Query: Governing Law          → columns: Document | Governing Law | Arbitration\n"
        "3. Use a markdown table with one row per document.\n"
        "4. Leave a cell blank or write 'Not found' if that attribute is absent in a document.\n\n"

        "## KEY DIFFERENCES\n"
        "- List only the 3–6 most significant differences.\n"
        "- Every bullet must reference at least one document and page.\n\n"

        "## SUMMARY\n"
        "Provide a single overall comparison in 2–4 sentences. Do not repeat document-by-document findings.\n"
    )


# ── Source extraction ─────────────────────────────────────────────────────────

def extract_sources(per_doc: Dict[str, List[Document]]) -> list:
    seen: set = set()
    sources = []
    for doc_id, chunks in per_doc.items():
        for chunk in chunks:
            source_file = os.path.basename(chunk.metadata.get("source", doc_id))
            page = chunk.metadata.get("page", "Unknown")
            key = (source_file, page)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": source_file,
                    "page": page,
                    "content_preview": chunk.page_content[:300] + "...",
                })
    sources.sort(key=lambda s: (s["file"], s["page"] if isinstance(s["page"], int) else 0))
    return sources


# ── SSE stream ────────────────────────────────────────────────────────────────

def stream_comparison(
    query: str,
    per_doc: Dict[str, List[Document]],
) -> Generator[str, None, None]:
    try:
        coverage = verify_coverage(per_doc)
        yield f"data: {json.dumps({'type': 'verification', 'coverage': coverage})}\n\n"

        prompt = build_comparison_prompt(query, per_doc, coverage)

        for chunk in llm.stream([HumanMessage(content=prompt)]):
            token = chunk.content
            if token:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        sources = extract_sources(per_doc)
        yield f"data: {json.dumps({'type': 'done', 'sources': sources})}\n\n"

    except Exception:
        logger.exception("compare/ failed for query=%r", query)
        yield f"data: {json.dumps({'type': 'error', 'detail': 'Something went wrong while comparing the documents. Please try again.'})}\n\n"
