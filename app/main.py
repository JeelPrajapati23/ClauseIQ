from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import json
import logging
import shutil
import os
from typing import List
import fitz  # PyMuPDF — already a transitive dep via pdf_parser

logger = logging.getLogger(__name__)

MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_PAGES = 150

# The system prompt asks the LLM for one exact refusal string, but llama-3.1-8b-instant
# often paraphrases it instead of reproducing it verbatim — match on the paraphrases too
# so sources still get wiped for a genuine refusal.
_REFUSAL_MARKERS = (
    "therefore an answer cannot be generated",
    "no information relevant",
    "does not contain information",
    "cannot answer this based on",
)

from app.loader import process_pdf
from app.database import save_chunks_to_vector_db, get_reranking_retriever, delete_user_document, attribute_answer_to_parents, QDRANT_URL
from pydantic import BaseModel, Field, field_validator
from app.generator import stream_answer, verify_answer_claims, rephrase_question, needs_rephrasing, classify_intent, QueryIntent
from groq import RateLimitError
from app.compare import retrieve_per_doc, stream_comparison

from app.auth.db import init_db, get_db
from app.auth.models import User, AuditLog
from app.auth.schemas import UserAdminItem, AuditLogItem
from app.auth.dependencies import require_active_user, require_admin
from app.auth.router import router as auth_router
from app.auth.utils import ENVIRONMENT
from app.rate_limit import limiter
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient

app = FastAPI(title="ClauseIQ RAG")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Comma-separated list of allowed frontend origins, e.g. "https://app.example.com,https://admin.example.com"
_DEFAULT_DEV_ORIGINS = "http://localhost:5173,http://localhost:5174,http://localhost:5175,http://localhost:5176"
CORS_ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", _DEFAULT_DEV_ORIGINS).split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # This is a JSON/SSE API with no server-rendered HTML, so a locked-down CSP is safe.
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    if ENVIRONMENT == "production":
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


app.include_router(auth_router)


@app.on_event("startup")
def on_startup():
    init_db()


UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _format_page_range(pages: list) -> str:
    """Merge consecutive page numbers into compact ranges: [3,4,5,7] → 'pp. 3-5, 7'"""
    if not pages:
        return ""
    try:
        nums = sorted(set(int(p) for p in pages))
    except (TypeError, ValueError):
        return ", ".join(str(p) for p in pages)
    ranges, start, end = [], nums[0], nums[0]
    for p in nums[1:]:
        if p == end + 1:
            end = p
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = p
    ranges.append(f"{start}-{end}" if start != end else str(start))
    prefix = "pp." if len(nums) > 1 else "p."
    return f"{prefix} {', '.join(ranges)}"


def _log(db: Session, action: str, user_id: str = None, detail: str = None, ip: str = None):
    db.add(AuditLog(user_id=user_id, action=action, detail=detail, ip_address=ip))
    db.commit()


def _safe_filename(filename: str) -> str:
    """Strip any path components so a crafted filename (e.g. '../../x.pdf') can't
    escape the per-user upload directory."""
    name = os.path.basename((filename or "").replace("\\", "/").strip())
    if not name or name in (".", "..") or "\x00" in name:
        raise HTTPException(400, "Invalid filename.")
    return name


@app.post("/upload-and-index/")
@limiter.limit("10/minute")
async def upload_and_process_pdf(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(require_active_user),
    db: Session = Depends(get_db),
):
    safe_filename = _safe_filename(file.filename)
    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are allowed.")
    user_dir = os.path.join(UPLOAD_DIR, current_user.id)
    os.makedirs(user_dir, exist_ok=True)
    file_path = os.path.join(user_dir, safe_filename)
    # Defense in depth: confirm the resolved path still lives under user_dir.
    if os.path.realpath(file_path) != os.path.join(os.path.realpath(user_dir), safe_filename):
        raise HTTPException(400, "Invalid filename.")
    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # ── Validate size ──────────────────────────────────────────────────
    actual_size = os.path.getsize(file_path)
    if actual_size > MAX_UPLOAD_BYTES:
        os.remove(file_path)
        raise HTTPException(
            400,
            f"File is {actual_size / (1024*1024):.1f} MB — maximum allowed is {MAX_UPLOAD_MB} MB.",
        )

    # ── Validate page count ────────────────────────────────────────────
    try:
        doc = fitz.open(file_path)
        page_count = len(doc)
        doc.close()
    except Exception:
        os.remove(file_path)
        raise HTTPException(400, "Could not read PDF. The file may be corrupted.")
    if page_count > MAX_PAGES:
        os.remove(file_path)
        raise HTTPException(
            400,
            f"PDF has {page_count} pages — maximum allowed is {MAX_PAGES} pages.",
        )

    try:
        chunks = process_pdf(file_path)
        save_chunks_to_vector_db(chunks, user_id=current_user.id)
        _log(db, "upload", user_id=current_user.id, detail=safe_filename)
        return {
            "filename": safe_filename,
            "status": "Successfully chunked and embedded",
            "total_chunks_saved": len(chunks),
        }
    except Exception:
        logger.exception("upload-and-index failed for user=%s file=%s", current_user.id, safe_filename)
        raise HTTPException(500, "Failed to process the uploaded document. Please try again.")


@app.delete("/documents/{source_file:path}")
async def delete_document(
    source_file: str,
    current_user: User = Depends(require_active_user),
    db: Session = Depends(get_db),
):
    """Remove all indexed chunks for one document owned by the current user."""
    safe_filename = _safe_filename(source_file)
    deleted = delete_user_document(source_file=safe_filename, user_id=current_user.id)
    if not deleted:
        raise HTTPException(404, f"Document '{safe_filename}' not found or already removed.")
    # Remove the physical file if it exists (source_file is sanitized to a bare
    # filename above, so this can't escape user_dir)
    user_dir = os.path.join(UPLOAD_DIR, current_user.id)
    physical = os.path.join(user_dir, safe_filename)
    if os.path.exists(physical):
        os.remove(physical)
    _log(db, "delete_document", user_id=current_user.id, detail=safe_filename)
    return {"deleted": safe_filename}


MAX_QUESTION_LEN = 4000
MAX_HISTORY_TURNS = 30
MAX_HISTORY_TURN_LEN = 4000
MAX_DOCUMENT_FILTER = 50


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LEN)
    history: list = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)
    document_filter: list = Field(default_factory=list, max_length=MAX_DOCUMENT_FILTER)

    @field_validator("history")
    @classmethod
    def _cap_turn_length(cls, turns):
        for turn in turns:
            content = turn.get("content", "") if isinstance(turn, dict) else str(turn)
            if len(str(content)) > MAX_HISTORY_TURN_LEN:
                raise ValueError(f"Each history turn's content must be at most {MAX_HISTORY_TURN_LEN} characters.")
        return turns


@app.post("/ask/")
@limiter.limit("20/minute")
async def ask_question(
    request: Request,
    body: AskRequest,
    current_user: User = Depends(require_active_user),
    db: Session = Depends(get_db),
):
    _log(db, "ask", user_id=current_user.id, detail=body.question[:200])

    def event_stream():
        try:
            intent = classify_intent(body.question)
            retriever = get_reranking_retriever(
                user_id=current_user.id,
                document_filter=body.document_filter or None,
                initial_k=20 if intent == QueryIntent.ANALYTICAL else 10,
                final_k=5 if intent == QueryIntent.ANALYTICAL else 3,
            )
            # Step 1 — resolve pronoun/reference ambiguity using conversation history
            retrieval_query = (
                rephrase_question(body.question, body.history)
                if body.history and needs_rephrasing(body.question)
                else body.question
            )
            retrieved_docs = retriever.invoke(retrieval_query)

            if not retrieved_docs:
                canned = "I cannot answer this based on the provided documents. No relevant context was found."
                yield f"data: {json.dumps({'type': 'token', 'content': canned})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
                yield f"data: {json.dumps({'type': 'verification', 'verdict': 'PASS', 'score': 1.0, 'total_claims': 0, 'unverified_claims': [], 'is_faithful': True})}\n\n"
                return

            # Build formatted context from retrieved parent docs
            context_parts = []
            for doc in retrieved_docs:
                sf = doc.metadata.get("source_file",
                     os.path.basename(doc.metadata.get("source", "Unknown")))
                all_pages = doc.metadata.get("all_pages") or [doc.metadata.get("page", "?")]
                section = doc.metadata.get("section", "")
                header = f"Source: {sf}, Pages: {_format_page_range(all_pages)}"
                if section:
                    header += f", Section: {section}"
                context_parts.append(f"{header}\nContent: {doc.page_content}\n")
            formatted_context = "\n---\n".join(context_parts)

            full_answer = ""
            for chunk in stream_answer(body.question, formatted_context, body.history, intent=intent):
                full_answer += chunk
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

            if any(marker in full_answer.lower() for marker in _REFUSAL_MARKERS):
                yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
                yield f"data: {json.dumps({'type': 'verification', 'verdict': 'PASS', 'score': 1.0, 'total_claims': 0, 'unverified_claims': [], 'is_faithful': True})}\n\n"
            else:
                # Attribute the answer to the parents that actually supported it
                attributed = attribute_answer_to_parents(full_answer, retrieved_docs)
                sources_metadata = []
                for doc in attributed:
                    sf = doc.metadata.get("source_file",
                         os.path.basename(doc.metadata.get("source", "Unknown")))
                    all_pages = doc.metadata.get("all_pages") or [doc.metadata.get("page")]
                    all_pages = [p for p in all_pages if p is not None]
                    section = doc.metadata.get("section", "")
                    sources_metadata.append({
                        "file": sf,
                        "pages": sorted(all_pages, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))),
                        "page_range": _format_page_range(all_pages),
                        "section": section,
                        "content_preview": doc.page_content[:300] + "...",
                    })
                sources_metadata.sort(key=lambda s: (s["file"], s["pages"][0] if s["pages"] else 0))

                yield f"data: {json.dumps({'type': 'done', 'sources': sources_metadata})}\n\n"
                report = verify_answer_claims(
                    question=body.question,
                    context=formatted_context,
                    generated_answer=full_answer,
                )
                unverified_claims = [c.claim_text for c in report.claims if not c.is_faithful]
                yield f"data: {json.dumps({'type': 'verification', 'verdict': report.verdict, 'score': report.faithfulness_score, 'total_claims': len(report.claims), 'unverified_claims': unverified_claims, 'is_faithful': report.verdict == 'PASS'})}\n\n"

        except RateLimitError:
            yield f"data: {json.dumps({'type': 'error', 'detail': 'The AI service is temporarily rate-limited. Please wait a few seconds and try again.'})}\n\n"
        except Exception:
            logger.exception("ask/ failed for user=%s", current_user.id)
            yield f"data: {json.dumps({'type': 'error', 'detail': 'Something went wrong while answering your question. Please try again.'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Compare Route ─────────────────────────────────────────────────────────────

MAX_COMPARE_DOCS = 20


class CompareRequest(BaseModel):
    doc_ids: list = Field(max_length=MAX_COMPARE_DOCS)
    query: str = Field(max_length=MAX_QUESTION_LEN)


@app.post("/compare/")
@limiter.limit("10/minute")
async def compare_documents(
    request: Request,
    body: CompareRequest,
    current_user: User = Depends(require_active_user),
    db: Session = Depends(get_db),
):
    if len(body.doc_ids) < 2:
        raise HTTPException(400, "At least 2 documents are required for comparison.")
    if not body.query.strip():
        raise HTTPException(400, "Query cannot be empty.")

    _log(db, "compare", user_id=current_user.id,
         detail=f"{len(body.doc_ids)} docs · {body.query[:120]}")

    per_doc = retrieve_per_doc(body.query, body.doc_ids, current_user.id)

    return StreamingResponse(
        stream_comparison(body.query, per_doc),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Admin Routes ──────────────────────────────────────────────────────────────

@app.get("/admin/users", response_model=List[UserAdminItem])
def admin_list_users(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return db.query(User).order_by(User.created_at.desc()).all()


@app.put("/admin/users/{user_id}/activate")
def admin_activate_user(
    user_id: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    user.is_active = True
    db.commit()
    _log(db, "admin_activate", user_id=admin.id, detail=f"activated {user.email}")
    return {"message": f"User {user.email} activated"}


@app.put("/admin/users/{user_id}/deactivate")
def admin_deactivate_user(
    user_id: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    if user.id == admin.id:
        raise HTTPException(400, "Cannot deactivate your own account")
    user.is_active = False
    db.commit()
    _log(db, "admin_deactivate", user_id=admin.id, detail=f"deactivated {user.email}")
    return {"message": f"User {user.email} deactivated"}


@app.get("/admin/audit-logs", response_model=List[AuditLogItem])
def admin_audit_logs(
    limit: int = 200,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit).all()


@app.get("/admin/documents")
def admin_list_documents(admin: User = Depends(require_admin)):
    try:
        client = QdrantClient(url=QDRANT_URL)
        records, _ = client.scroll(
            collection_name="pdf_knowledge_base", limit=5000, with_payload=True
        )
        seen = set()
        docs = []
        for r in records:
            meta = r.payload.get("metadata", {})
            uid = meta.get("user_id")
            sf = meta.get("source_file")
            if uid and sf and (uid, sf) not in seen:
                seen.add((uid, sf))
                docs.append({"user_id": uid, "source_file": sf})
        return docs
    except Exception:
        logger.exception("admin/documents scan failed")
        raise HTTPException(500, "Failed to list documents.")
