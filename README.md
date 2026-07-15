# ClauseIQ — Multi-Tenant RAG for Legal Documents

**Live**: [clauseiq-rag.vercel.app](https://clauseiq-rag.vercel.app)

A full-stack Retrieval-Augmented Generation (RAG) system for question-answering and cross-document comparison over legal PDFs (contracts, agreements, licenses). Each user gets an isolated, authenticated workspace — upload your own documents, ask questions grounded strictly in their content, or compare multiple contracts side by side, with every answer checked for faithfulness before it reaches the user.

Built as an end-to-end system: FastAPI backend, Qdrant hybrid search + Cohere reranking, Postgres-backed auth, a React chat UI, Docker orchestration, a deployed production stack with CI/CD, and an offline Ragas evaluation harness — not just a notebook demo.

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
- [Production Deployment](#production-deployment)
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
- JWT-cookie authentication (httponly, 8h expiry) backed by Postgres, with password reset and per-account login lockout after repeated failures.
- Every document and every Qdrant vector is scoped to `user_id` — no cross-user data access anywhere in ask/compare/delete.
- **Chat history persists server-side in Postgres**, not the browser — conversations, citations, and verification results follow the account across devices/browsers/logouts instead of living in `localStorage`.
- Admin panel: user activation/deactivation, audit log viewer, cross-tenant document inventory.
- Full audit trail (`audit_logs` table) for register/login/logout/upload/ask/compare/delete/admin actions.
- Rate limiting (per-IP) and per-account lockout as independent defenses against brute force.

**Engineering**
- Dockerized end-to-end (backend, frontend, Qdrant, Postgres) via a single `docker compose up`.
- Deployed to production: Vercel (frontend) + Azure Container Apps (backend) + Neon (Postgres) + Qdrant Cloud, with GitHub Actions CI/CD redeploying the backend on every push to `main`.
- Frontend and backend on different domains are made same-origin via a Vercel rewrite proxy (`/api/*` → Azure), so the auth cookie survives strict cross-site tracking protections (Brave Shields, Safari ITP, Firefox strict mode) without a token-in-header rewrite.
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
        Postgres[("Postgres<br/>users, audit_logs, chat history")]
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

**Request flow for `/ask`:**

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
| Deployment | Vercel (frontend, with a rewrite proxy to the backend), Azure Container Apps (backend), Neon (Postgres), Qdrant Cloud |
| CI/CD | GitHub Actions — builds and pushes the backend image, updates the Container App on every push to `main` |

---

## Project Structure

```
ask-my-docs-rag/
├── app/
│   ├── main.py                 # FastAPI routes: upload, ask (SSE), compare (SSE), delete, chat sessions, admin
│   ├── pdf_parser.py           # PyMuPDF extraction, two-column detection, reading-order sort
│   ├── loader.py               # Parent-child chunking (2000/0 parents, 350/50 children)
│   ├── database.py             # Qdrant client, BM25 cache, hybrid retriever, reranker, attribution
│   ├── generator.py            # Groq streaming, intent classification, faithfulness verification
│   ├── compare.py               # Multi-document MMR retrieval + comparison prompt + SSE
│   ├── batch_index.py           # Standalone bulk-indexer for ingestion-docs/ (full collection wipe)
│   ├── create_qdrant_indexes.py # One-off backfill: Qdrant Cloud payload indexes on a pre-existing collection
│   ├── rate_limit.py             # Shared slowapi Limiter, keyed by real client IP (X-Forwarded-For aware)
│   └── auth/
│       ├── models.py            # User, AuditLog, DocumentJob, ChatSession, ChatMessage SQLAlchemy models
│       ├── db.py                # Postgres engine (pool_pre_ping for Neon), migrations
│       ├── router.py            # register/login/logout/forgot/reset/change-password
│       ├── dependencies.py      # require_active_user, require_admin guards
│       ├── utils.py             # bcrypt + JWT helpers, ENVIRONMENT flag
│       └── email_utils.py       # SMTP or console-log dev fallback
├── prompts/                      # Versioned system prompts (v1/v2/v3, analytical)
├── Frontend/
│   ├── src/
│   │   ├── chat.jsx             # Main chat/compare UI
│   │   ├── App.jsx              # Auth-gated root
│   │   ├── components/          # Auth, AdminPanel, FileChip, ResetPasswordModal, icons
│   │   └── utils/                # api.js (API_BASE_URL), sessionsApi.js (chat session/message CRUD), storage.js (pending-upload-job tracking only)
│   ├── vercel.json               # Rewrites /api/* to the Azure backend (same-origin proxy)
│   └── Dockerfile               # Multi-stage Vite build -> nginx
├── deploy/azure/
│   ├── bootstrap.sh              # One-time az CLI provisioning of the prod backend
│   └── .env.azure.example        # Template for bootstrap.sh's secrets (real file is gitignored)
├── .github/workflows/
│   └── deploy-backend.yml        # CI/CD: builds + deploys the backend on push to main
├── Evaluation/
│   ├── golden_qa_set.json        # Intent-tagged golden Q&A set (factual/analytical/out-of-scope)
│   ├── golden_doc_map.json       # source_row -> originating contract map
│   ├── evaluate_rag_offline.py   # Runs the real pipeline per question, scores with Ragas
│   ├── ragas_eval_results_scoped.csv   # Current, final results (66 rows)
│   └── build_golden_set.py, build_golden_doc_map.py, ...  # Golden set generation scripts
├── ingestion-docs/               # Sample legal PDFs for batch_index.py
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
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/clauseiq
APP_URL=http://localhost:8000                  # backend origin (also baked into the frontend build as VITE_API_URL — unused in production, see below)
APP_FRONTEND_URL=http://localhost:5173         # used to build password-reset redirect links
CORS_ALLOWED_ORIGINS=http://localhost:5173,http://localhost:5174
QDRANT_URL=http://localhost:6333               # or a Qdrant Cloud cluster URL in production
QDRANT_API_KEY=                                # required by Qdrant Cloud; leave unset for a local instance

# Optional — LangSmith tracing (off if unset)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=clauseiq-rag

# Optional — Email (omit to print password-reset links to console instead)
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASS=
FROM_EMAIL=noreply@clauseiq.local
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
docker run -d --name clauseiq-postgres -p 5433:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=clauseiq postgres:16-alpine
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

1. Register an account with just an email and password (the first-ever user becomes admin).
2. Upload one or more PDFs — indexing happens immediately on file select.
3. **Ask** questions grounded in your uploaded documents; answers stream in with cited sources and a faithfulness verdict.
4. Select 2+ documents and switch to **Compare** mode to get a structured, per-document analysis of the same question.
5. Admins can access `/admin` to manage users and view audit logs.

---

## Production Deployment

The live app runs across four separate managed services rather than a single host:

| Service | Provider | Notes |
|---|---|---|
| Frontend | Vercel | Auto-deployed from `main` via Vercel's GitHub integration |
| Backend | Azure Container Apps | `min-replicas=0` to conserve free/student credit — first request after idle is a cold start |
| Relational DB | Neon (serverless Postgres) | `users`, `audit_logs`, `document_jobs`, `chat_sessions`, `chat_messages` |
| Vector DB | Qdrant Cloud | `pdf_knowledge_base` collection |

**Same-origin cookie auth across two domains.** The frontend (`clauseiq-rag.vercel.app`) and backend (an Azure Container Apps FQDN) are on different domains, which normally forces the auth cookie to be `SameSite=None; Secure` — and strict cross-site tracking protections (Brave Shields, Safari ITP, Firefox strict mode) block `SameSite=None` cookies outright, regardless of `Secure`. Instead of switching to a token-in-header scheme, `Frontend/vercel.json` proxies `/api/*` through to the Azure backend, so the browser sees every API call as same-origin — letting the cookie use `SameSite=Lax`, which those browsers don't block. Vercel's external rewrite is a genuine reverse-proxy pass-through (not the older, buffered Vercel Functions streaming model), but it does impose a hard **120-second timeout** on any single proxied request — worth knowing if you extend `/compare/` to cover many large documents at once.

**CI/CD.** `.github/workflows/deploy-backend.yml` runs on every push to `main` that touches backend code: it builds the Docker image locally on the GitHub-hosted runner (Azure's ACR Tasks build service is disabled on free/student subscriptions), pushes it to Azure Container Registry, and updates the running Container App to the new image. The frontend has no separate workflow — Vercel deploys it independently on every push. One-time infrastructure provisioning (resource group, registry, Container Apps environment) lives in `deploy/azure/bootstrap.sh` and is idempotent.

**Two managed-cloud gotchas worth knowing if you fork this:**
- **Neon closes idle Postgres connections server-side.** Without `pool_pre_ping=True` on the SQLAlchemy engine, the first query after any idle period fails with `OperationalError: SSL connection has been closed unexpectedly` instead of transparently reconnecting.
- **Qdrant Cloud rejects filtered queries on a field with no payload index** (`Index required but not found`), unlike a local/unauthenticated Qdrant instance. `ensure_payload_indexes()` creates the required indexes on every write; `app/create_qdrant_indexes.py` backfills them on a collection that already had data before that fix existed.

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
- httponly JWT cookies (no token-in-header/localStorage flow), CSP + security headers on every response, HSTS in production. `SameSite=Lax` in production, made possible by proxying the frontend and backend to the same origin (see [Production Deployment](#production-deployment)) rather than relying on `SameSite=None`, which strict tracking-protection browsers block outright.
- Per-IP rate limiting (`slowapi`) **and** independent per-account login lockout after repeated failures. The rate limiter and audit log both resolve the real client IP from `X-Forwarded-For` rather than the raw TCP peer — necessary because a reverse-proxied deployment (Vercel → Azure Container Apps ingress) otherwise makes every request appear to originate from the same internal ingress IP, collapsing per-IP limits into one shared, global limit.
- Uploaded/deleted filenames are sanitized to a bare basename before touching the filesystem — no path traversal.
- Request bodies are size-capped (question/history length, filter list sizes) to bound LLM cost and context abuse.
- Unexpected server errors are logged internally and never leak raw exception details to the client.
- Full audit log of every meaningful account and data action, including real client IP.

---

## Design Decisions

- **Two-tier parent-child chunking** — retrieval precision needs small chunks; legal answers need full clause context. Splitting the two lets both be true at once.
- **Two-call, claim-level faithfulness verification** — a single "is this faithful, yes/no" LLM call is too coarse and self-serving; extracting claims *before* showing context (to avoid bias) and checking each one individually against retrieved text gives a much more granular, auditable faithfulness score.
- **Intent-based routing over one-size-fits-all retrieval** — analytical/advisory questions need more supporting context than narrow factual lookups; a keyword classifier (not an extra LLM call) makes this routing free.
- **Incremental per-user indexing vs. full bulk reindex** — the live upload path only touches the affected user's `(user_id, source_file)` pair; the standalone `batch_index.py` script is intentionally separate and destructive, reserved for offline corpus loading.
- **Same-origin proxy over token-in-header auth** — cross-domain deployment (Vercel + Azure) usually pushes people toward moving the JWT out of a cookie and into an `Authorization` header to sidestep `SameSite` entirely. Instead, a Vercel rewrite proxy makes the two origins look identical to the browser, keeping the simpler httponly-cookie model and its XSS-exfiltration resistance, at the cost of a 120-second timeout on any single proxied request.
- **Proxy-aware client IP over trusting `request.client.host`** — any reverse-proxied deployment (here: Vercel → Azure Container Apps ingress) makes the TCP-level peer address useless for per-visitor rate limiting or audit attribution, since it's always the last proxy's own IP. Reading the leftmost `X-Forwarded-For` entry fixes the common case; it's knowingly spoofable by a request sent directly to the backend's own public FQDN, which is an accepted tradeoff at this app's scale rather than a fully hop-counted, spoof-proof implementation.
