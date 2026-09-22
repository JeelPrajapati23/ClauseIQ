import os
import re
import time
from typing import Any, List
from dotenv import load_dotenv
from cohere.errors.too_many_requests_error import TooManyRequestsError
from langchain_cohere import CohereRerank
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Self-contained rather than relying on import order, so the embeddings client
# below always has its API token regardless of which module imports this first.
load_dotenv()
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue, PointIdsList, PayloadSchemaType
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from pydantic import ConfigDict

# Served via HF's Inference Providers router using HUGGINGFACEHUB_API_TOKEN.
# This model is asymmetric: queries need a "query: " prefix, documents are
# embedded plain (per the model card), so embed_query is overridden below.
class _ArcticLegalEmbeddings(HuggingFaceEndpointEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([f"query: {text}"])[0]

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([f"query: {text}"]))[0]


embeddings = _ArcticLegalEmbeddings(
    model="Snowflake/snowflake-arctic-embed-l-v2.0",
    task="feature-extraction",
    model_kwargs={"normalize": True},
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # required by Qdrant Cloud; unset for a local/unauthenticated instance

# Every field ever passed to a Qdrant FieldCondition filter (user_id/source_file scoping
# on save, delete, retrieval and compare). A local/unauthenticated Qdrant instance filters
# on unindexed fields without complaint, but Qdrant Cloud rejects the query outright with
# "Index required but not found" unless a payload index exists for the field.
_FILTERED_PAYLOAD_FIELDS = ("metadata.user_id", "metadata.source_file")


def ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """
    Create a keyword payload index for every field this app ever filters on.
    Safe to call repeatedly: Qdrant treats re-creating an index with the same
    field/schema as a no-op, and any unexpected error is swallowed so this
    never blocks the write path it's called from.
    """
    for field_name in _FILTERED_PAYLOAD_FIELDS:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


# rank_bm25's Okapi scoring stays as-is; only the tokenization feeding it changes.
# The library default (BM25Retriever's default_preprocessing_func) is a bare
# text.split() — no lowercasing, no punctuation handling — so "Section" vs "section"
# mismatch, and "3.1(b)." becomes one indivisible token fused to its own punctuation.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "as", "by", "with", "from", "such", "which", "who",
    "whom", "if", "then", "than", "so", "into", "onto", "upon", "under", "over",
    "between", "within", "any", "all", "each", "other",
})
# Deliberately NOT stopworded: shall/may/must/will/not/no — these carry real legal
# meaning (obligation vs. permission, negation), unlike generic function words above.
_BM25_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'\-][a-z0-9]+)*")


def bm25_preprocess(text: str) -> list[str]:
    """Lowercase and tokenize on word boundaries, keeping section references like
    '3.1' and hyphenated/possessive terms like 'co-branding'/'lessee's' as single
    tokens instead of losing them to naive whitespace splitting, then drop
    low-signal function-word stopwords. Applied identically to both indexed
    documents and queries (BM25Retriever calls this same function on each).
    """
    tokens = _BM25_TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


# Per-user BM25 cache: user_id -> {retriever, version, collection, k, filter}
_bm25_version: int = 0
_user_bm25_cache: dict = {}


def invalidate_bm25_cache() -> None:
    global _bm25_version
    _bm25_version += 1


def _build_bm25_retriever(collection_name: str, k: int, doc_filter: tuple, user_id: str):
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        user_filter = Filter(must=[FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))])
        records, _ = client.scroll(
            collection_name=collection_name,
            limit=2000,
            with_payload=True,
            scroll_filter=user_filter,
        )
        if not records:
            return None
        docs = [
            Document(
                page_content=r.payload.get("page_content", ""),
                metadata=r.payload.get("metadata", {}),
            )
            for r in records
        ]
        if doc_filter:
            docs = [d for d in docs if d.metadata.get("source_file", "") in doc_filter]
        if not docs:
            return None
        retriever = BM25Retriever.from_documents(docs, preprocess_func=bm25_preprocess)
        retriever.k = k
        return retriever
    except Exception:
        return None


def _get_cached_bm25_retriever(collection_name: str, k: int, doc_filter: tuple, user_id: str):
    global _user_bm25_cache, _bm25_version
    cached = _user_bm25_cache.get(user_id, {})
    if (
        cached.get("version") != _bm25_version
        or cached.get("collection") != collection_name
        or cached.get("k") != k
        or cached.get("filter") != doc_filter
    ):
        _user_bm25_cache[user_id] = {
            "retriever": _build_bm25_retriever(collection_name, k, doc_filter, user_id),
            "version": _bm25_version,
            "collection": collection_name,
            "k": k,
            "filter": doc_filter,
        }
    return _user_bm25_cache[user_id]["retriever"]


def _delete_points_by_filter(client: QdrantClient, collection_name: str, filt: Filter) -> int:
    """
    Scroll to collect matching point IDs, then delete them explicitly.
    Passing a bare Filter as points_selector is unreliable in qdrant-client ≥1.x;
    delete-by-IDs is the stable path across all versions.
    Returns the number of points deleted.
    """
    try:
        records, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=filt,
            limit=10_000,
            with_payload=False,
            with_vectors=False,
        )
        if not records:
            return 0
        ids = [r.id for r in records]
        client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=ids),
        )
        return len(ids)
    except Exception:
        return 0


def save_chunks_to_vector_db(chunks, user_id: str, collection_name="pdf_knowledge_base"):
    """Stamps user_id onto every chunk, deletes existing user-owned points for the same file, then upserts."""
    for chunk in chunks:
        chunk.metadata["user_id"] = user_id

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    # Must run before the delete-by-filter below too, in case the collection
    # predates these payload indexes.
    ensure_payload_indexes(client, collection_name)
    source_files = {c.metadata.get("source_file") for c in chunks if c.metadata.get("source_file")}
    for sf in source_files:
        _delete_points_by_filter(
            client,
            collection_name,
            Filter(must=[
                FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="metadata.source_file", match=MatchValue(value=sf)),
            ]),
        )

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name,
        force_recreate=False,
        check_compatibility=False,
    )
    # Re-run after from_documents() too, since a first-ever upload creates the
    # collection here rather than it existing for the call above.
    ensure_payload_indexes(client, collection_name)
    invalidate_bm25_cache()


def swap_to_parent_context(child_docs: list) -> list:
    """
    Replace each child's page_content with its parent_context, collecting every
    contributing page number into metadata["all_pages"] for merged citations.
    Legacy chunks without parent_context pass through unchanged.
    """
    parents: dict = {}
    legacy: list = []
    for doc in child_docs:
        parent_text = doc.metadata.get("parent_context")
        if not parent_text:
            legacy.append(doc)
            continue
        key = doc.metadata.get("parent_id") or parent_text[:80]
        if key not in parents:
            meta = {k: v for k, v in doc.metadata.items() if k != "parent_context"}
            meta["all_pages"] = []
            parents[key] = Document(page_content=parent_text, metadata=meta)
        page = doc.metadata.get("page")
        if page is not None and page not in parents[key].metadata["all_pages"]:
            parents[key].metadata["all_pages"].append(page)
    return list(parents.values()) + legacy


def get_full_document_context(
    user_id: str, source_file: str, collection_name: str = "pdf_knowledge_base"
) -> list:
    """Returns every unique parent chunk for one document, bypassing retrieval entirely.

    Used for clause-category questions ("does this have a non-compete clause?"),
    where the category name and the clause's actual wording can share little
    vocabulary — handing the LLM the whole document sidesteps relevance ranking
    instead of relying on it to find the right chunk.

    Scoped to a single already-uploaded document, not a multi-doc or unscoped
    search — see the caller for how document_filter is checked before this path
    is used.
    """
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    doc_filter = Filter(must=[
        FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id)),
        FieldCondition(key="metadata.source_file", match=MatchValue(value=source_file)),
    ])
    records, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=doc_filter,
        limit=2000,
        with_payload=True,
    )
    if not records:
        return []
    child_docs = [
        Document(page_content=r.payload.get("page_content", ""), metadata=r.payload.get("metadata", {}))
        for r in records
    ]
    return swap_to_parent_context(child_docs)


def attribute_answer_to_parents(answer: str, parent_docs: list, threshold: float = 0.3) -> list:
    """
    Return the subset of parent_docs most semantically similar to the generated answer.
    Uses cosine similarity via the shared embedding model (embeddings are L2-normalised,
    so dot product == cosine). Falls back to all parents if anything goes wrong.
    """
    if not parent_docs:
        return parent_docs
    try:
        import numpy as np
        vecs = np.array(embeddings.embed_documents(
            [answer] + [doc.page_content for doc in parent_docs]
        ))
        scores = vecs[1:] @ vecs[0]          # cosine similarity of each parent to answer
        attributed = [doc for doc, s in zip(parent_docs, scores) if s >= threshold]
        return attributed if attributed else [parent_docs[int(np.argmax(scores))]]
    except Exception:
        return parent_docs


def delete_user_document(source_file: str, user_id: str, collection_name="pdf_knowledge_base") -> int:
    """
    Delete every Qdrant point that belongs to *user_id* and *source_file*.
    Returns the number of points deleted (0 means document not found or error).
    """
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    deleted = _delete_points_by_filter(
        client,
        collection_name,
        Filter(must=[
            FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="metadata.source_file", match=MatchValue(value=source_file)),
        ]),
    )
    if deleted:
        invalidate_bm25_cache()
    return deleted


def query_vector_db(query: str, k: int = 4, collection_name: str = "pdf_knowledge_base"):
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        check_compatibility=False,
    )
    return qdrant.similarity_search(query=query, k=k)


_RERANK_MAX_RETRIES = 3
_RERANK_BASE_DELAY = 5.0  # seconds


def _rerank_with_retry(compressor, context_docs: list, query: str) -> list:
    """Cohere trial keys carry a short per-minute burst limit on top of the
    monthly call cap (both surface as the same 429 TooManyRequestsError, with
    the response headers distinguishing which — x-trial-endpoint-call-remaining
    near-exhausted means the burst limit, not necessarily the monthly one). A
    short retry with backoff rides out the burst-limit case instead of failing
    the whole /ask/ or /compare/ request on a transient rate limit."""
    for attempt in range(_RERANK_MAX_RETRIES):
        try:
            return compressor.compress_documents(context_docs, query)
        except TooManyRequestsError:
            if attempt == _RERANK_MAX_RETRIES - 1:
                raise
            time.sleep(_RERANK_BASE_DELAY * (2 ** attempt))
    return []


class ThresholdReranker(BaseRetriever):
    """Hybrid retriever: Qdrant vector + BM25 fused via RRF, then Cohere cross-encoder reranked.

    Cohere's relevance scores don't separate relevant from irrelevant documents
    reliably enough for an absolute cutoff, so out-of-scope questions are instead
    caught downstream by the LLM's own refusal string (see main.py's guardrail check).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ensemble_retriever: Any
    compressor: Any

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        child_docs = self.ensemble_retriever.invoke(query)
        # Swap to parent context before reranking so Cohere scores full legal
        # sections, not the tiny child fragments used for retrieval.
        context_docs = swap_to_parent_context(child_docs)
        return _rerank_with_retry(self.compressor, context_docs, query)


def get_reranking_retriever(
    user_id: str,
    collection_name="pdf_knowledge_base",
    initial_k=10,
    final_k=3,
    document_filter=None,
):
    """Hybrid retriever scoped to a single user via Qdrant payload filter."""
    doc_filter = tuple(sorted(document_filter)) if document_filter else ()

    filter_conditions = [FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))]
    if doc_filter:
        filter_conditions.append(
            FieldCondition(key="metadata.source_file", match=MatchAny(any=list(doc_filter)))
        )
    qdrant_filter = Filter(must=filter_conditions)

    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        check_compatibility=False,
    )
    vector_retriever = qdrant.as_retriever(search_kwargs={"k": initial_k, "filter": qdrant_filter})

    bm25_retriever = _get_cached_bm25_retriever(collection_name, initial_k, doc_filter, user_id)

    if bm25_retriever is not None:
        # Fusion weights barely matter here: Cohere reranking below re-scores the
        # full candidate pool from scratch, regardless of fusion order.
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )
    else:
        ensemble_retriever = vector_retriever

    compressor = CohereRerank(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        model="rerank-english-v3.0",
        top_n=final_k,
    )

    return ThresholdReranker(
        ensemble_retriever=ensemble_retriever,
        compressor=compressor,
    )
