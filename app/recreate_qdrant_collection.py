"""
One-off fix for a Qdrant collection left stale by an embedding-model swap.

Deletes the named collection outright. The next real document upload (via
app/database.py:save_chunks_to_vector_db -> QdrantVectorStore.from_documents)
auto-recreates it from scratch with whatever dimension/distance the CURRENT
embeddings model (app/database.py's `embeddings`) actually produces, plus the
usual payload indexes (ensure_payload_indexes runs on every write already).

Use this when upload-and-index is failing with:
    QdrantVectorStoreError: Existing Qdrant collection is configured for dense
    vectors with N dimensions. Selected embeddings are M-dimensional.
i.e. the collection predates a real embedding-model change (different vector
space, not just a hosting swap) and needs a full reindex.

DESTRUCTIVE: every point in the collection is deleted. Every user's previously
indexed documents become unsearchable until they re-upload — there is no
server-side original to silently re-embed (uploaded_files/{user_id}/ is
write-once-then-read-once and Render's filesystem is ephemeral). Confirm this
is actually wanted before running against production.

Usage:
    QDRANT_URL=... QDRANT_API_KEY=... python app/recreate_qdrant_collection.py [collection_name] [--yes]

collection_name defaults to "pdf_knowledge_base" (the production collection).
Without --yes, prints what it would do and exits without deleting anything.
"""
import os
import sys

from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def main():
    args = [a for a in sys.argv[1:] if a != "--yes"]
    confirmed = "--yes" in sys.argv
    collection_name = args[0] if args else "pdf_knowledge_base"

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Collection: {collection_name}")

    try:
        info = client.get_collection(collection_name)
        points_count = info.points_count
        print(f"Current points: {points_count}")
    except Exception as exc:
        print(f"Could not read collection info (it may not exist): {exc}")
        points_count = None

    if not confirmed:
        print(
            "\nDRY RUN — no changes made. This would permanently delete the "
            f"'{collection_name}' collection and all {points_count if points_count is not None else '?'} "
            "points in it. Every user's previously uploaded documents would stop "
            "being searchable until they re-upload.\n"
            "Re-run with --yes to actually delete it."
        )
        return

    client.delete_collection(collection_name)
    print(f"\nDeleted collection '{collection_name}'. It will be recreated "
          "automatically, with the current embeddings model's dimensions, on "
          "the next document upload.")


if __name__ == "__main__":
    main()
