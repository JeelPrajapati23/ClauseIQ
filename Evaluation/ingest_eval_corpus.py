"""
Bulk-ingest Evaluation/ingestion_pdfs/ into a dedicated, disposable Qdrant
collection used only for evaluation — isolated from the production
`pdf_knowledge_base` collection so this script can never wipe real user data.

Every chunk is stamped with a fixed EVAL_USER_ID so the production
get_reranking_retriever() (which filters by metadata.user_id) can be pointed
at this collection unmodified.

Usage:
    python Evaluation/ingest_eval_corpus.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Unlike running through uvicorn (where main.py/generator.py load .env first), this
# script is a standalone entry point — without this, HUGGINGFACEHUB_API_TOKEN (and
# every other .env var app.database/app.loader rely on) is silently unset, and HF's
# embeddings API rejects the unauthenticated request with a 401 that looks like a
# bad/misscoped token but isn't.
load_dotenv(PROJECT_ROOT / ".env")

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.database import embeddings, QDRANT_URL
from app.loader import process_pdf

PDF_DIR = Path(__file__).resolve().parent / "ingestion_pdfs"
COLLECTION_NAME = os.getenv("EVAL_COLLECTION_NAME", "eval_knowledge_base")
EVAL_USER_ID = os.getenv("EVAL_USER_ID", "eval")


def index_eval_corpus() -> None:
    if not PDF_DIR.exists():
        print(f"[ERROR] Directory not found: {PDF_DIR}")
        return

    # Case-insensitive: Linux (WSL2's /mnt/c included) treats *.pdf and *.PDF as
    # different globs even on a case-preserving Windows-backed filesystem.
    pdf_files = sorted(
        p for p in PDF_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"
    )
    if not pdf_files:
        print(f"[WARN] No PDF files found in {PDF_DIR}")
        return

    print(f"[INFO] Found {len(pdf_files)} PDFs in {PDF_DIR}")

    all_chunks = []
    for idx, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{idx}/{len(pdf_files)}] Processing: {pdf_path.name}")
        try:
            chunks = process_pdf(str(pdf_path))
            for chunk in chunks:
                chunk.metadata["user_id"] = EVAL_USER_ID
            all_chunks.extend(chunks)
            print(f"    [OK] Generated {len(chunks)} chunks")
        except Exception as e:
            print(f"    [ERROR] {pdf_path.name}: {e}")

    if not all_chunks:
        print("[ABORT] No chunks extracted")
        return

    print(f"\n[INFO] Total chunks generated: {len(all_chunks)}")
    print(f"[INFO] Recreating Qdrant collection: {COLLECTION_NAME}")

    try:
        QdrantVectorStore.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
            force_recreate=True,
            check_compatibility=False,
        )

        client = QdrantClient(url=QDRANT_URL)
        collection_info = client.get_collection(COLLECTION_NAME)

        print("\n========== COLLECTION INFO ==========")
        print(collection_info)
        print(
            f"\n[SUCCESS] Indexed {len(pdf_files)} PDFs / {len(all_chunks)} chunks "
            f"into '{COLLECTION_NAME}' under user_id='{EVAL_USER_ID}'."
        )
    except Exception as e:
        print(f"\n[ERROR] Qdrant indexing failed:\n{e}")


if __name__ == "__main__":
    print("=" * 60)
    print("EVALUATION CORPUS INGESTION")
    print("=" * 60)
    index_eval_corpus()
