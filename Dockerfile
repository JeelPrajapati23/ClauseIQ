FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# No torch/transformers/sentence-transformers here — the embedding model in
# app/database.py runs via the HF Inference API, not in-process. Those heavy
# deps only exist in requirements-batch.txt, for the standalone local
# app/batch_index.py bulk-loading script, which never runs in this image.
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY prompts/ ./prompts/

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p uploaded_files logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Single worker: the BM25 retriever cache in app/database.py is per-process
# in-memory and invalidated on write within that same process. Running
# multiple workers/replicas would let one process write while another keeps
# serving a stale cache — don't scale this service horizontally without
# moving that cache to a shared store (e.g. Redis) first.
#
# Shell form (not exec-form JSON array) so $PORT actually expands — Render
# assigns its own PORT at runtime and expects the app to bind to it; falls
# back to 8000 for docker-compose/Cloud Run, which don't set it.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
