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

---

## Key Features

**Retrieval & Generation**
- **Hybrid search** — Qdrant dense vector search fused with BM25 (Reciprocal Rank Fusion), reranked by Cohere `rerank-english-v3.0`.
- **Parent-child chunking** — small 350-char children carry precise embeddings for retrieval; 2000-char parent sections are swapped back in at answer time so the LLM sees full legal-clause context, not fragments.
- **Intent-aware routing** — questions are classified `FACT` vs `ANALYTICAL` (keyword-based, no extra LLM call) and routed to different retrieval widths and system prompts.
- **Conversational rephrasing** — follow-up questions with pronouns ("what about *that* clause?") are rewritten into standalone queries using conversation history before retrieval, only when needed.
- **Two-call claim-level faithfulness verification** — after every answer, a separate LLM pass extracts each factual claim (without seeing the context, to avoid bias) and checks it against the retrieved chunks for a verbatim supporting quote, producing a `PASS` / `PARTIAL` / `FAIL` verdict and a 0–1 faithfulness score per answer.
- **Streaming answers** — token-by-token via Server-Sent Events (SSE), with sources and verification results streamed alongside.
- **Multi-document comparison mode** — ask one question across 2+ documents; the model is instructed to evaluate each document independently and cite file + page for every claim, never mixing facts across documents.
- **Trained off-topic classifier** — every question first passes a "System-1" gate: a logistic-regression classifier over the question's embedding, trained on ~10k labelled questions, returns a calibrated off-topic probability and decides ~90% of questions on its own in milliseconds. Only the uncertain ~10% go to an LLM. Replacing the per-question LLM check cut wrongly-refused legal questions from 37% to 1.6% on a held-out sample and roughly halved Gemini usage per question (see [Evaluation Results](#off-topic-classifier)).
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
- Deployed to production: Vercel (frontend) + Render (backend) + Neon (Postgres) + Qdrant Cloud, auto-deploying on every push to `main`.
- Frontend and backend on different domains are made same-origin via a Vercel rewrite proxy, so the auth cookie survives strict cross-site tracking protections (Brave Shields, Safari ITP, Firefox strict mode) without a token-in-header rewrite.
- Offline evaluation harness (Ragas, judged by a separate Groq-hosted model) that exercises the *real* production pipeline against a golden question set derived from the CUAD legal-contracts dataset.

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
        Ask["Ask (SSE)<br/>off-topic gate -> intent classify -> retrieve -> generate -> verify"]
        Gate["Off-topic gate<br/>keywords -> embedding classifier -> LLM only if unsure"]
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
        Gemini["Gemini 2.5 Flash — LLM generation<br/>(rotating multi-key)"]
        Groq["Groq — claim<br/>verification only"]
        CohereAPI["Cohere — reranking"]
        Embed["Hugging Face — hosted<br/>embeddings API"]
    end

    UI <--> |"JWT httponly cookie"| Auth
    UI --> Upload
    UI --> Ask
    UI --> Compare
    UI --> Admin

    Upload --> Embed --> Qdrant
    Upload --> Files

    Ask --> Gate
    Compare --> Gate
    Gate --> |"question embedding"| Embed
    Gate -.-> |"uncertain ~10%"| Gemini
    Ask --> Hybrid
    Compare --> Hybrid
    Hybrid --> Qdrant
    Hybrid --> Rerank --> CohereAPI
    Hybrid --> ParentSwap
    ParentSwap --> Gemini
    Gemini --> |"faithfulness check"| Groq

    Auth --> Postgres
    Admin --> Postgres
    Admin --> Qdrant
```

**Request flow for `/ask`:**

1. Off-topic gate: a keyword list, then the trained embedding classifier (block / allow / unsure), then a Gemini check only for unsure questions. Off-topic questions get a canned refusal before any retrieval. The question's embedding is cached, so retrieval reuses it.
2. Classify intent (`FACT` / `ANALYTICAL`) → picks retrieval width + system prompt.
3. If the question references prior turns ("it", "that", …), rephrase into a standalone query using conversation history.
4. Hybrid retrieve (vector + BM25 → RRF) scoped to `user_id`, rerank with Cohere, swap children for parent context.
5. If nothing relevant was retrieved → stream a canned refusal, skip the LLM entirely.
6. Stream the answer token-by-token from Gemini 2.5 Flash (a rotating pool of free-tier API keys — see [Tech Stack](#tech-stack)).
7. Attribute the answer back to source parents by embedding similarity (cosine ≥ 0.3).
8. Run the two-call claim-level faithfulness audit and stream the verdict + sources.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI, Uvicorn |
| LLM generation | Gemini 2.5 Flash, streamed via SSE, behind a rotating multi-key client (`app/gemini_client.py`) to pool free-tier daily quotas — prototype-scale traffic only, not a substitute for a paid tier |
| Structured extraction | `instructor`-patched Groq client (claim extraction/verification — deliberately a different model than generation, so it never self-grades) |
| Off-topic classifier | Logistic regression on query embeddings, trained with a numpy Newton solver and served with numpy alone (`app/offtopic_classifier.py`, weights in `app/models/`) |
| Vector database | Qdrant |
| Reranking | Cohere `rerank-english-v3.0` |
| Embeddings | `Snowflake/snowflake-arctic-embed-l-v2.0`, served via Hugging Face's hosted Inference API |
| Keyword search | BM25 (rank_bm25), per-user in-memory cache |
| Relational DB | PostgreSQL + SQLAlchemy |
| Auth | JWT (`python-jose`), `bcrypt` password hashing, httponly cookies |
| PDF parsing | PyMuPDF, with two-column layout detection and reading-order correction |
| Rate limiting | `slowapi` |
| Tracing (optional) | LangSmith |
| Frontend | React 19, Vite, Tailwind CSS, `react-markdown` |
| Evaluation | Ragas, judged offline by a separate Groq-hosted model |
| Containerization | Docker, Docker Compose |
| Deployment | Vercel (frontend, with a rewrite proxy to the backend), Render (backend), Neon (Postgres), Qdrant Cloud |
| CI/CD | Vercel and Render each auto-deploy from `main` |

---

## Project Structure

```
ask-my-docs-rag/
├── app/
│   ├── main.py                 # FastAPI routes: upload, ask (SSE), compare (SSE), delete, chat sessions, admin
│   ├── pdf_parser.py           # PyMuPDF extraction, two-column detection, reading-order sort
│   ├── loader.py               # Parent-child chunking (2000/0 parents, 350/50 children)
│   ├── database.py             # Qdrant client, BM25 cache, hybrid retriever, reranker, attribution
│   ├── generator.py            # Gemini streaming, off-topic gate, intent classification, faithfulness verification
│   ├── offtopic_classifier.py  # Trained off-topic classifier (scores the question embedding, numpy only)
│   ├── models/                 # offtopic_classifier.json — exported classifier weights + thresholds
│   ├── gemini_client.py        # Rotating multi-key Gemini client (production generation model)
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
├── prompts/                      # Versioned system prompts (fact + analytical)
├── Frontend/
│   ├── src/
│   │   ├── chat.jsx             # Main chat/compare UI
│   │   ├── App.jsx              # Auth-gated root
│   │   ├── components/          # Auth, AdminPanel, FileChip, ResetPasswordModal, icons
│   │   └── utils/                # api.js (API_BASE_URL), sessionsApi.js (chat session/message CRUD), storage.js (pending-upload-job tracking only)
│   ├── vercel.json               # Rewrites /api/* to the backend (same-origin proxy)
│   └── Dockerfile               # Multi-stage Vite build -> nginx
├── render.yaml                    # Render Blueprint: provisions the production backend service
├── Evaluation/
│   ├── golden_qa_set.json        # Intent-tagged golden Q&A set (factual/analytical/out-of-scope)
│   ├── golden_doc_map.json       # source_row -> originating contract map
│   ├── evaluate_rag_offline.py   # Runs the real pipeline per question, scores with Ragas
│   ├── sweep_retrieval_params.py # Fast, judge-free retrieval parameter sweep
│   ├── offtopic/                 # Off-topic classifier: dataset building, labelling guide, training, comparison
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
- API keys: [Google AI Studio](https://aistudio.google.com/apikey) (generation), [Groq](https://console.groq.com/) (claim verification), [Cohere](https://dashboard.cohere.com/), [Hugging Face](https://huggingface.co/settings/tokens)

### Environment Variables

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=...                            # generation model. Comma-separated for multiple keys/projects —
                                               # Google AI Studio's free tier caps each project at 20 requests/day,
                                               # so app/gemini_client.py rotates across keys instead of failing
GROQ_API_KEY=...                              # used for claim verification (app/generator.py's _verifier) and the eval judge, not generation
COHERE_API_KEY=...
HUGGINGFACEHUB_API_TOKEN=...                  # embeddings, via HF's hosted Inference API
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
| Backend | Render | Auto-deployed from `main`, provisioned via `render.yaml` |
| Relational DB | Neon (serverless Postgres) | `users`, `audit_logs`, `document_jobs`, `chat_sessions`, `chat_messages` |
| Vector DB | Qdrant Cloud | `pdf_knowledge_base` collection |

**Same-origin cookie auth across two domains.** The frontend and backend live on different domains, which normally forces the auth cookie to be `SameSite=None; Secure` — and strict cross-site tracking protections (Brave Shields, Safari ITP, Firefox strict mode) block `SameSite=None` cookies outright, regardless of `Secure`. Instead of switching to a token-in-header scheme, `Frontend/vercel.json` proxies `/api/*` through to the backend, so the browser sees every API call as same-origin — letting the cookie use `SameSite=Lax`, which those browsers don't block. Vercel's external rewrite is a genuine reverse-proxy pass-through, but it does impose a hard **120-second timeout** on any single proxied request — worth knowing if you extend `/compare/` to cover many large documents at once.

**Two managed-cloud gotchas worth knowing if you fork this:**
- **Neon closes idle Postgres connections server-side.** Without `pool_pre_ping=True` on the SQLAlchemy engine, the first query after any idle period fails instead of transparently reconnecting.
- **Qdrant Cloud rejects filtered queries on a field with no payload index**, unlike a local/unauthenticated Qdrant instance. `ensure_payload_indexes()` creates the required indexes on every write; `app/create_qdrant_indexes.py` backfills them on a collection that already had data before that fix existed.

---

## Evaluation Results

The eval harness runs the **actual production pipeline** (same retriever/generator code as the live API) against a golden question set derived from [CUAD](https://www.atticusprojectai.org/cuad) (real-world commercial legal contracts), scored offline via Ragas with a separate Groq-hosted judge model — the judge never grades its own generations.

**Overall (68 questions), current production model (Gemini 2.5 Flash):**

| Metric | Score |
|---|---|
| Faithfulness | 0.89 |
| Answer Relevancy | 0.69 |
| Context Precision | 0.66 |
| Context Recall | 0.83 |

**By question type (faithfulness):**

| Intent | n | Faithfulness |
|---|---|---|
| Factual | 44 | 0.85 |
| Analytical | 18 | 0.97 |
| Out-of-scope (guardrail) | 6 | 1.00 |

**Notes:**
- All out-of-scope questions correctly triggered the refusal guardrail (faithfulness 1.0 by policy — a refusal cannot be unfaithful).
- Retrieval parameters (`initial_k`/`final_k`) and the retriever/prompt code are unchanged from the prior model — this run isolates the effect of the generation-model swap (`ragas_eval_results_gemini.csv`, vs. the prior `openai/gpt-oss-20b` baseline in `ragas_eval_results_gptoss.csv`: faithfulness 0.68, answer relevancy 0.69, context precision 0.59, context recall 0.69, factual faithfulness 0.57). The gap is driven by faithfulness and context recall specifically — Gemini reproduces retrieved figures/quotes more literally, which the claim-entailment scorer rewards directly.
- Gemini generation runs behind a rotating multi-key client (`app/gemini_client.py`) to work around Google AI Studio's free-tier cap of 20 requests/day per project — sound at this app's prototype traffic level, not at production scale without a paid tier.

Reproduce locally:
```bash
cd Evaluation
python evaluate_rag_offline.py   # resumable — skips source_rows already in the output CSV
```

### Off-topic classifier

The off-topic gate used to send every question to Gemini with a yes/no prompt. That cost half of each question's free-tier quota, added ~2 s, and, as measured below, refused a large share of legitimate legal questions. It's now a trained classifier with Gemini only as a fallback.

**How it was built** (`Evaluation/offtopic/`):
- **Boundary first.** A written labelling guide (`LABELLING_GUIDE.md`, 10 on-topic and 9 off-topic rules) defines what counts. Notably, general legal questions count as on-topic, and the guide leans toward on-topic because wrongly refusing a real user costs more than one wasted LLM call.
- **Data.** ~10.7k questions:
  - 6.8k from public datasets: PrivacyQA, LegalBench consumer-contracts QA and Law Stack Exchange (on-topic and grey-zone), CLINC150 and English WildChat (off-topic). Licences and samples were verified live.
  - ~4k generated: contract questions from real CUAD clauses in varied styles (casual, typos, fragments, follow-ups, pasted clauses), plus hard negatives that look on-topic but aren't.
- **Labels.** `gpt-oss-120b` labelled every row without seeing its source; rows where its label disagreed with the source's were re-checked.
- **Leakage controls.** Splits are grouped so rewordings of one question never cross train/test, and near-duplicates (e.g. one WildChat prompt template repeated across conversations) are removed from train.
- **Model.** Logistic regression on the question's `snowflake-arctic-embed-l-v2.0` embedding: the same embedding retrieval already computes, so the gate adds no extra API call when the question isn't rephrased. Regularisation and two decision thresholds are set on validation data (block if ≤1% of on-topic questions would be blocked; allow if ≤2% of off-topic would get through). Questions in between go to Gemini.

**Held-out test set (422 questions, including every hard case from the test pool):**

| Metric | Score |
|---|---|
| On-topic questions wrongly blocked | 1.1% (3/263; 95% CI 0.4–3.3%) |
| Off-topic questions let through | 1.3% (2/159; 95% CI 0.4–4.5%) |
| Sent to the Gemini fallback | 10% |
| ROC AUC | 0.993 |
| Calibration error (ECE) | 0.026 |

**Old gate vs. new, same 100-question stratified sample of the test set:**

| | Old: Gemini on every question | New: classifier + Gemini when unsure |
|---|---|---|
| On-topic questions wrongly blocked | **37%** (23/62) | **1.6%** (1/62) |
| Off-topic questions let through | 2.6% (1/38) | 0% (0/38) |
| Accuracy | 76% | 99% |
| Gemini calls per question (gate only) | 1.0 | 0.09 |
| Gate latency | ~2 s median (Gemini call) | ~0.3 s (embedding, reused by retrieval) |

About a third of the old gate's 23 false blocks come from the guide deliberately widening the boundary (general legal questions are now on-topic). The rest were core usage the old prompt got wrong by its own definition: pasted clauses, negotiation questions, privacy-policy questions.

**Caveats:**
- The test set's labels were checked by Claude (a second model family reviewing `gpt-oss-120b`'s labels, overriding 1.4%), not by a human. These numbers measure agreement with two independent models applying the guide.
- A second round of training data targeted categories found by inspecting test errors (questions about the app or the conversation, mixed requests). That made the final test numbers mildly optimistic.
- Known gap: a mixed request ("explain the payment schedule *and* set a reminder") can still be blocked, because one embedding of the whole message is dominated by its off-topic part.
- The gate is a **cost and quality filter, not a security boundary**: wrapping any request in legal vocabulary scores on-topic. User isolation is enforced by per-user filters in the data layer, not by the gate (see [Security Highlights](#security-highlights)).

Reproduce (each step is resumable; see the docstrings in `Evaluation/offtopic/`):
```bash
python Evaluation/offtopic/fetch_sources.py
python Evaluation/offtopic/generate_questions.py
python Evaluation/offtopic/generate_targeted.py
python Evaluation/offtopic/label_with_llm.py
python Evaluation/offtopic/build_splits.py
python Evaluation/offtopic/finalize_test_set.py --reviewer <who reviewed test_to_review.csv>
python Evaluation/offtopic/embed_dataset.py
python Evaluation/offtopic/train_classifier.py
python Evaluation/offtopic/compare_classifiers.py --dry-run   # uses live Gemini quota; check the call count first
```

---

## Security Highlights

- Per-user data isolation enforced at the Qdrant filter level on every retrieval, index, and delete — no cross-tenant access path exists.
- httponly JWT cookies (no token-in-header/localStorage flow), CSP + security headers on every response, HSTS in production. `SameSite=Lax` in production, made possible by proxying the frontend and backend to the same origin (see [Production Deployment](#production-deployment)) rather than relying on `SameSite=None`, which strict tracking-protection browsers block outright.
- Per-IP rate limiting (`slowapi`) **and** independent per-account login lockout after repeated failures. The rate limiter and audit log both resolve the real client IP from `X-Forwarded-For` rather than the raw TCP peer, since a reverse-proxied deployment would otherwise attribute every request to the same proxy IP.
- Uploaded/deleted filenames are sanitized to a bare basename before touching the filesystem — no path traversal.
- Request bodies are size-capped (question/history length, filter list sizes) to bound LLM cost and context abuse.
- The off-topic gate's LLM fallback treats the user's question as data: it's passed as an escaped JSON string inside delimiters, and only an exact `ON_TOPIC`/`OFF_TOPIC` reply is accepted. The gate itself is documented as a cost/quality filter, not a security control.
- Unexpected server errors are logged internally and never leak raw exception details to the client.
- Full audit log of every meaningful account and data action, including real client IP.
