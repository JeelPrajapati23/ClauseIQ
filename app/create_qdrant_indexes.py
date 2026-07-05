"""
One-off/retroactive fix for collections created before payload indexes were added
to the write path (see app/database.py:ensure_payload_indexes). Qdrant Cloud rejects
filtered queries on a field with no payload index ("Index required but not found"),
unlike a local/unauthenticated Qdrant instance, which filters on unindexed fields
without complaint.

Run once against any existing collection that predates that fix — safe to re-run,
Qdrant treats re-creating an index with the same field/schema as a no-op.

Usage:
    QDRANT_URL=... QDRANT_API_KEY=... python app/create_qdrant_indexes.py [collection_name]

collection_name defaults to "pdf_knowledge_base" (the production collection).
"""
import os
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
FIELDS = ("metadata.user_id", "metadata.source_file")


def main():
    collection_name = sys.argv[1] if len(sys.argv) > 1 else "pdf_knowledge_base"
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    print(f"Collection: {collection_name}")
    print(f"Qdrant URL: {QDRANT_URL}")

    for field_name in FIELDS:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print(f"  [OK] indexed {field_name}")

    info = client.get_collection(collection_name)
    print("\nPayload schema now:")
    print(info.payload_schema)


if __name__ == "__main__":
    main()
