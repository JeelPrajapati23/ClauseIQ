# Ask My Docs — Multi-Tenant RAG for Legal Documents

A full-stack Retrieval-Augmented Generation (RAG) system for question-answering and cross-document comparison over legal PDFs (contracts, agreements, licenses). Each user gets an isolated, authenticated workspace — upload your own documents, ask questions grounded strictly in their content, or compare multiple contracts side by side, with every answer checked for faithfulness before it reaches the user.

Built as an end-to-end system: FastAPI backend, Qdrant hybrid search + Cohere reranking, Postgres-backed auth, a React chat UI, Docker orchestration, and an offline Ragas evaluation harness — not just a notebook demo.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Run with Docker Compose](#run-with-docker-compose-recommended)
  - [Run Locally (without Docker)](#run-locally-without-docker)
- [Using the App](#using-the-app)
- [Evaluation Results](#evaluation-results)
- [Security Highlights](#security-highlights)
- [Design Decisions](#design-decisions)

---

## Key Features

**Retrieval & Generation**
- **Hybrid search** — Qdrant dense vector search fused with BM25 (Reciprocal Rank Fusion), reranked by Cohere `rerank-english-v3.0`.
- **Parent-child chunking** — small 350-char children carry precise embeddings for retrieval; 2000-char parent sections are swapped back in at answer time so the LLM sees full legal-clause context, not fragments.
- **Intent-aware routing** — questions are classified `FACT` vs `ANALYTICAL` (keyword-based, no extra LLM call) and routed to different retrieval widths (`k=3` vs `k=5`) and system prompts.
- **Conversational rephrasing** — follow-up questions with pronouns ("what about *that* clause?") are rewritten into standalone queries using conversation history before retrieval, only when needed.
- **Two-call claim-level faithfulness verification** — after every answer, a separate LLM pass extracts each factual claim (without seeing the context, to avoid bias) and checks it against the retrieved chunks for a verbatim supporting quote, producing a `PASS` / `PARTIAL` / `FAIL` verdict and a 0–1 faithfulness score per answer.
- **Streaming answers** — token-by-token via Server-Sent Events (SSE), with sources and verification results streamed alongside.
- **Multi-document comparison mode** — ask one question across 2+ documents; the model is instructed to evaluate each document independently and cite file + page for every claim, never mixing facts across documents.
- **Guardrails** — if no relevant context is retrieved, a canned refusal is streamed with zero LLM calls; refusal-paraphrase detection prevents a "confident-sounding non-answer" from being mis-verified as faithful.

**Multi-Tenancy & Auth**
- JWT-cookie authentication (httponly, 8h expiry) backed by Postgres, with email verification, password reset, and per-account login lockout after repeated failures.
- Every document and every Qdrant vector is scoped to `user_id` — no cross-user data access anywhere in ask/compare/delete.
- Admin panel: user activation/deactivation, audit log viewer, cross-tenant document inventory.
- Full audit trail (`audit_logs` table) for register/login/logout/upload/ask/compare/delete/admin actions.
- Rate limiting (per-IP) and per-account lockout as independent defenses against brute force.
- A shared, hourly-reset **public demo account** so reviewers can try the app without registering.

**Engineering**
- Dockerized end-to-end (backend, frontend, Qdrant, Postgres) via a single `docker compose up`.
- Offline evaluation harness (Ragas, judged locally via Ollama) that exercises the *real* production pipeline against a golden question set derived from the CUAD legal-contracts dataset.

---

## Architecture

```mermaid
flowchart TB
    subgraph Client["Browser"]
        UI["React / Vite Chat UI<br/>(chat, compare, admin panel)"]
    end

    subgraph Backend["FastAPI Backend"]
        Auth["Auth Router<br/>JWT cookie · lockout · audit log"]
        Upload["Upload & Index<br/>PDF parse -> chunk -> embed"]
        Ask["Ask (SSE)<br/>intent classify -> retrieve -> generate -> verify"]
        Compare["Compare (SSE)<br/>per-doc MMR retrieval -> multi-doc prompt"]
        Admin["Admin Routes<br/>users · audit logs · documents"]
    end

    subgraph Retrieval["Retrieval Layer"]
        Hybrid["Hybrid Retriever<br/>Qdrant vector search + BM25 -> RRF fusion"]
        Rerank["Cohere rerank-english-v3.0"]
        ParentSwap["Parent-context swap"]
    end

    subgraph Data["Data Stores"]
        Qdrant[("Qdrant<br/>vectors, per-user filtered")]
        Postgres[("Postgres<br/>users, audit_logs")]
        Files[("uploaded_files/{user_id}/")]
    end

    subgraph External["External APIs"]
        Groq["Groq — llama-3.1-8b-instant<br/>(generation + claim verification)"]
        CohereAPI["Cohere — reranking"]
        BGE["BAAI/bge-base-en-v1.5<br/>(local embeddings, CPU)"]
    end

    UI <--> |"JWT httponly cookie"| Auth
    UI --> Upload
    UI --> Ask
    UI --> Compare
    UI --> Admin

    Upload --> BGE --> Qdrant
    Upload --> Files

    Ask --> Hybrid
    Compare --> Hybrid
    Hybrid --> Qdrant
    Hybrid --> Rerank --> CohereAPI
    Hybrid --> ParentSwap
    ParentSwap --> Groq
    Groq --> |"faithfulness check"| Groq

    Auth --> Postgres
    Admin --> Postgres
    Admin --> Qdrant
```

**Request flow for `/ask/`:**

1. Classify intent (`FACT` / `ANALYTICAL`) → picks retrieval width + system prompt.
2. If the question references prior turns ("it", "that", …), rephrase into a standalone query using conversation history.
3. Hybrid retrieve (vector + BM25 → RRF) scoped to `user_id`, rerank with Cohere, swap children for parent context.
4. If nothing relevant was retrieved → stream a canned refusal, skip the LLM entirely.
5. Stream the answer token-by-token from Groq.
6. Attribute the answer back to source parents by embedding similarity (cosine ≥ 0.3).
7. Run the two-call claim-level faithfulness audit and stream the verdict + sources.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI, Uvicorn |
| LLM generation | Groq (`llama-3.1-8b-instant`), streamed via SSE |
| Structured extraction | `instructor`-patched Groq client (claim extraction/verification) |
| Vector database | Qdrant |
| Reranking | Cohere `rerank-english-v3.0` |
| Embeddings | `BAAI/bge-base-en-v1.5` (sentence-transformers, CPU in the API, CUDA-optional for bulk ingest) |
| Keyword search | BM25 (rank_bm25), per-user in-memory cache |
| Relational DB | PostgreSQL + SQLAlchemy |
| Auth | JWT (`python-jose`), `bcrypt` password hashing, httponly cookies |
| PDF parsing | PyMuPDF, with two-column layout detection and reading-order correction |
| Rate limiting | `slowapi` |
| Tracing (optional) | LangSmith |
| Frontend | React 19, Vite, Tailwind CSS, `react-markdown` |
| Evaluation | Ragas, judged offline by a local Ollama model (`llama3.1:8b`) |
| Containerization | Docker, Docker Compose |

---

## Project Structure

```
ask-my-docs-rag/
├── app/
│   ├── main.py                 # FastAPI routes: upload, ask (SSE), compare (SSE), delete, admin
│   ├── pdf_parser.py           # PyMuPDF extraction, two-column detection, reading-order sort
│   ├── loader.py               # Parent-child chunking (2000/0 parents, 350/50 children)
│   ├── database.py             # Qdrant client, BM25 cache, hybrid retriever, reranker, attribution
│   ├── generator.py            # Groq streaming, intent classification, faithfulness verification
│   ├── compare.py               # Multi-document MMR retrieval + comparison prompt + SSE
│   ├── batch_index.py           # Standalone bulk-indexer for ingestion-docs/ (full collection wipe)
│   ├── rate_limit.py             # Shared slowapi Limiter
│   └── auth/
│       ├── models.py            # User, AuditLog SQLAlchemy models
│       ├── db.py                # Postgres engine, migrations, demo-account seeding
│       ├── router.py            # register/login/logout/verify/forgot/reset/change-password
│       ├── dependencies.py      # require_active_verified, require_admin guards
│       ├── utils.py             # bcrypt + JWT helpers, ENVIRONMENT flag
│       └── email_utils.py       # SMTP or console-log dev fallback
├── prompts/                      # Versioned system prompts (v1/v2/v3, analytical)
├── Frontend/
│   ├── src/
│   │   ├── chat.jsx             # Main chat/compare UI
│   │   ├── App.jsx              # Auth-gated root
│   │   ├── components/          # Auth, AdminPanel, FileChip, ResetPasswordModal, icons
│   │   └── utils/                # api.js (API_BASE_URL), storage.js
│   └── Dockerfile               # Multi-stage Vite build -> nginx
├── Evaluation/
│   ├── golden_qa_set.json        # Intent-tagged golden Q&A set (factual/analytical/out-of-scope)
│   ├── golden_doc_map.json       # source_row -> originating contract map
│   ├── evaluate_rag_offline.py   # Runs the real pipeline per question, scores with Ragas
│   ├── ragas_eval_results_scoped.csv   # Current, final results (66 rows)
│   └── build_golden_set.py, build_golden_doc_map.py, ...  # Golden set generation scripts
├── ingestion-docs/               # Sample legal PDFs for batch_index.py
├── .github/workflows/reset-demo.yml   # Hourly cron resetting the shared demo account
├── Dockerfile                     # Backend image (python:3.11-slim, CPU-only torch)
├── docker-compose.yml             # Backend + frontend + Qdrant + Postgres
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the frontend)
- Docker (for Qdrant/Postgres, or the whole stack)
- API keys: [Groq](https://console.groq.com/), [Cohere](https://dashboard.cohere.com/)

### Environment Variables

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=...
COHERE_API_KEY=...
RAG_SYSTEM_PROMPT_FILE=system_prompt_v3.txt   # optional, defaults to v3

# Auth
ENVIRONMENT=development                        # "production" enforces JWT_SECRET_KEY + secure cookies
JWT_SECRET_KEY=...                             # required in production
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/askmydocs
APP_URL=http://localhost:8000                  # used to build email-verification links
APP_FRONTEND_URL=http://localhost:5173         # used to build password-reset redirect links
CORS_ALLOWED_ORIGINS=http://localhost:5173,http://localhost:5174
QDRANT_URL=http://localhost:6333

# Optional — LangSmith tracing (off if unset)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=ask-my-docs-rag

# Optional — Email (omit to print verification/reset links to console instead)
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASS=
FROM_EMAIL=noreply@askmydocs.local

# Optional — demo account hourly reset endpoint
DEMO_RESET_SECRET=...
```

### Run with Docker Compose (recommended)

```bash
docker compose up --build
```

- Backend → `http://localhost:8000`
- Frontend → `http://localhost:5173`
- Qdrant → `http://localhost:6333`
- Postgres → `localhost:5433`

> The frontend image bakes `VITE_API_URL` in at **build time** — rebuild the frontend image if the backend URL changes.

### Run Locally (without Docker)

**1. Start Qdrant and Postgres:**
```bash
docker run -p 6333:6333 qdrant/qdrant
docker run -d --name askmydocs-postgres -p 5433:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=askmydocs postgres:16-alpine
```

**2. Backend:**
```bash
python -m venv myenv
myenv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
# API at http://localhost:8000
```

**3. Frontend:**
```bash
cd Frontend
npm install
npm run dev
# UI at http://localhost:5173
```

**Optional — bulk-index a folder of PDFs** (bypasses auth/user-scoping, wipes the collection — for offline corpus loading, not per-user uploads):
```bash
python app/batch_index.py
```

---

## Using the App

1. Register an account (first-ever user is auto-verified as admin) or click **Log in as Guest / Demo**.
2. Upload one or more PDFs — indexing happens immediately on file select.
3. **Ask** questions grounded in your uploaded documents; answers stream in with cited sources and a faithfulness verdict.
4. Select 2+ documents and switch to **Compare** mode to get a structured, per-document analysis of the same question.
5. Admins can access `/admin` to manage users and view audit logs.

---

## Evaluation Results

The eval harness runs the **actual production pipeline** (same retriever/generator code as the live API) against a golden question set derived from [CUAD](https://www.atticusprojectai.org/cuad) (real-world commercial legal contracts), scored offline via Ragas with a local Ollama judge (`llama3.1:8b`) — no eval traffic hits Groq/Cohere for scoring.

**Overall (66 questions, `ragas_eval_results_scoped.csv`):**

| Metric | Score |
|---|---|
| Faithfulness | 0.71 |
| Answer Relevancy | 0.82 |
| Context Precision | 0.85 |

**By question type:**

| Intent | n | Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|---|---|
| Factual | 42 | 0.62 | 0.82 | 0.81 |
| Analytical | 18 | 0.85 | 0.80 | 0.96 |
| Out-of-scope (guardrail) | 6 | 1.00 | — | — |

**Notes:**
- All 6 out-of-scope questions correctly triggered the refusal guardrail (faithfulness 1.0 by policy — a refusal cannot be unfaithful).
- Analytical questions score *higher* on faithfulness than factual ones here — counter-intuitive at first glance, but consistent with wider retrieval (`k=5` vs `k=3`) giving the analytical prompt more supporting context per claim.
- Factual faithfulness (0.62) is pulled down by strict claim-level entailment scoring from a local 8B judge model — row-level inspection found several low-scoring answers that were in fact near-verbatim-correct; this reflects judge harshness on claim decomposition as much as retrieval quality.
- A previous pipeline iteration (`ragas_eval_results_v3.csv`, fact-only golden set, no intent routing) scored ~0.86 faithfulness — **not directly comparable**, since the current golden set intentionally adds harder analytical/advisory questions and a stricter two-call verification method.

Reproduce locally:
```bash
cd Evaluation
python evaluate_rag_offline.py   # resumable — skips source_rows already in the output CSV
```

---

## Security Highlights

- Per-user data isolation enforced at the Qdrant filter level on every retrieval, index, and delete — no cross-tenant access path exists.
- httponly JWT cookies (no token-in-header/localStorage flow), CSP + security headers on every response, HSTS in production.
- Per-IP rate limiting (`slowapi`) **and** independent per-account login lockout after repeated failures.
- Uploaded/deleted filenames are sanitized to a bare basename before touching the filesystem — no path traversal.
- Request bodies are size-capped (question/history length, filter list sizes) to bound LLM cost and context abuse.
- Unexpected server errors are logged internally and never leak raw exception details to the client.
- Full audit log of every meaningful account and data action.

---

## Design Decisions

- **Two-tier parent-child chunking** — retrieval precision needs small chunks; legal answers need full clause context. Splitting the two lets both be true at once.
- **Two-call, claim-level faithfulness verification** — a single "is this faithful, yes/no" LLM call is too coarse and self-serving; extracting claims *before* showing context (to avoid bias) and checking each one individually against retrieved text gives a much more granular, auditable faithfulness score.
- **Intent-based routing over one-size-fits-all retrieval** — analytical/advisory questions need more supporting context than narrow factual lookups; a keyword classifier (not an extra LLM call) makes this routing free.
- **Incremental per-user indexing vs. full bulk reindex** — the live upload path only touches the affected user's `(user_id, source_file)` pair; the standalone `batch_index.py` script is intentionally separate and destructive, reserved for offline corpus loading.
- **Shared, auto-resetting demo account** — lets reviewers try the full authenticated experience without registering, while a scheduled `/demo/reset` endpoint (run in-process, not a standalone script) keeps it clean and prevents any one visitor from locking it out for everyone else.
