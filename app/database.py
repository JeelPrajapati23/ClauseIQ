import os
from typing import Any, List
from langchain_cohere import CohereRerank
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue, PointIdsList, PayloadSchemaType
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from pydantic import ConfigDict

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
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
        retriever = BM25Retriever.from_documents(docs)
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
    # Must run before the delete-by-filter below, not just after from_documents() —
    # on a collection that predates this index (e.g. an existing prod collection until
    # the one-off backfill below has run), the delete filter needs it too.
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
    # Also re-run after from_documents(): on a first-ever upload the collection didn't
    # exist for the call above, so this is what actually creates the index for it.
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


class ThresholdReranker(BaseRetriever):
    """Hybrid retriever: Qdrant vector + BM25 fused via RRF, then Cohere cross-encoder reranked.

    Cohere's raw relevance scores for long legal-document/instructional-query pairs sit in a
    noise floor (~0.0001-0.05) that overlaps between genuinely relevant and irrelevant documents,
    so no absolute score cutoff can separate them reliably. Out-of-scope questions are instead
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
        return self.compressor.compress_documents(context_docs, query)


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
