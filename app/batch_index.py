import os
import shutil
from pathlib import Path

import torch
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from app.loader import process_pdf

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "./ingestion-docs"
UPLOAD_DIR = "./uploaded_files"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "pdf_knowledge_base"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================
# EMBEDDINGS
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Using device: {device}")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={
        "device": device
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)

# ============================================================
# INDEXING
# ============================================================

def index_all_documents():

    source_dir = Path(DATA_DIR)

    if not source_dir.exists():
        print(f"[ERROR] Directory not found: {DATA_DIR}")
        return

    pdf_files = list(source_dir.rglob("*.pdf"))

    if not pdf_files:
        print(f"[WARN] No PDF files found in {DATA_DIR}")
        return

    print(f"[INFO] Found {len(pdf_files)} PDFs")

    all_chunks = []

    for idx, pdf_path in enumerate(pdf_files, start=1):

        print(
            f"[{idx}/{len(pdf_files)}] Processing: {pdf_path.name}"
        )

        try:

            backup_path = os.path.join(
                UPLOAD_DIR,
                pdf_path.name
            )

            shutil.copyfile(pdf_path, backup_path)

            chunks = process_pdf(str(pdf_path))

            for chunk in chunks:
                chunk.metadata["source_file"] = pdf_path.name

            all_chunks.extend(chunks)

            print(
                f"    [OK] Generated {len(chunks)} chunks"
            )

        except Exception as e:
            print(
                f"    [ERROR] {pdf_path.name}: {str(e)}"
            )

    if not all_chunks:
        print("[ABORT] No chunks extracted")
        return

    print(
        f"\n[INFO] Total chunks generated: {len(all_chunks)}"
    )

    print(
        f"[INFO] Recreating Qdrant collection: {COLLECTION_NAME}"
    )

    try:

        QdrantVectorStore.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
            force_recreate=True,
            check_compatibility=False
        )

        print(
            "\n[SUCCESS] Documents indexed successfully."
        )

        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        collection_info = client.get_collection(
            COLLECTION_NAME
        )

        print("\n========== COLLECTION INFO ==========")
        print(collection_info)

    except Exception as e:
        print(
            f"\n[ERROR] Qdrant indexing failed:\n{e}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LEGAL DOCUMENT REINDEXING")
    print("=" * 60)

    if torch.cuda.is_available():
        print(
            f"[INFO] GPU Detected: {torch.cuda.get_device_name(0)}"
        )
    else:
        print("[INFO] Running on CPU")

    index_all_documents()

