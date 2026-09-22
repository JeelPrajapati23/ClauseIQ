import os
import re
import sys
import time
import logging
from enum import Enum
from typing import Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import instructor
from groq import Groq, RateLimitError

load_dotenv()

# Generation model. The internal claim verifier (_verifier below) and the Ragas
# judge intentionally stay on a different model than whatever wins this slot, so
# neither self-grades its own generations.
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    reasoning_effort="low",
    api_key=os.getenv("GROQ_API_KEY")
)

_MAX_RETRIES = 3
_BASE_DELAY = 2.0  # seconds

def _parse_retry_after(exc: RateLimitError) -> float:
    """Extract the suggested wait time from the Groq error message, fall back to base delay."""
    match = re.search(r"try again in (\d+\.?\d*)s", str(exc))
    return float(match.group(1)) if match else _BASE_DELAY

def _invoke_with_retry(func, *args, **kwargs):
    """Calls func(*args, **kwargs) and retries on Groq rate-limit errors with exponential backoff."""
    for attempt in range(_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except RateLimitError as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_parse_retry_after(exc) * (2 ** attempt))


class QueryIntent(Enum):
    FACT = "fact"
    ANALYTICAL = "analytical"


_ANALYTICAL_KEYWORDS = {
    "explain", "summarize", "summary", "compare", "contrast", "describe",
    "discuss", "analyze", "analyse", "overview", "outline", "elaborate",
    "walk me through", "tell me about", "give me a", "give an", "provide a",
    "what are all", "list all", "enumerate", "breakdown", "break down",
    "how does", "how do", "in detail", "comprehensively", "thoroughly",
    "in depth", "deep dive", "what is the significance", "cover all",
}

def classify_intent(question: str) -> QueryIntent:
    """Classifies the question as ANALYTICAL or FACT based on keyword signals."""
    q_lower = question.lower()
    if any(kw in q_lower for kw in _ANALYTICAL_KEYWORDS):
        return QueryIntent.ANALYTICAL
    return QueryIntent.FACT


# Qdrant/BM25 retrieval always returns its top-k nearest chunks regardless of true
# relevance (see ThresholdReranker's docstring in app/database.py), so a question with
# zero relationship to the uploaded documents — e.g. a request for code — still reaches
# the LLM with legal-document context attached. The system prompt asks the LLM to refuse
# in that case, but a small instruction-following model can still answer a strongly
# in-distribution request (like reversing a linked list) instead of refusing. This keyword
# gate catches the obvious cases deterministically, before any retrieval or LLM call.
_OFF_TOPIC_KEYWORDS = {
    "write a python", "write python", "python function", "python code", "python script",
    "javascript function", "javascript code", "write a function", "write code",
    "write a program", "write an algorithm", "leetcode", "reverse a linked list",
    "linked list", "binary search tree", "binary tree", "sort an array",
    "fibonacci sequence", "sql query", "regex for", "regular expression for",
    "write a regex", "html code", "css code", "for loop", "while loop",
    "write a poem", "write an essay", "write a story", "write a song",
    "recipe for", "joke about", "tell me a joke", "translate this to",
    "capital of france", "write a sql",
}


def is_off_topic_request(question: str) -> bool:
    """Cheap deterministic gate for requests unrelated to legal-document QA (code
    generation, creative writing, general trivia) that retrieval alone can't filter out."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in _OFF_TOPIC_KEYWORDS)


# Backstop for the keyword gate above, which can't enumerate every off-topic
# phrasing: one fast classification call, before any retrieval, so an off-topic
# question never reaches the document-QA prompt regardless of phrasing.
_OFF_TOPIC_CLASSIFIER_PROMPT = """You are a strict binary classifier guarding a legal-document Q&A assistant.

The assistant may ONLY answer questions that could plausibly be answered by looking inside a user-uploaded legal document (contracts, agreements, clauses, parties, obligations, dates, payment terms, termination conditions, definitions, etc.), including meta-questions about the conversation itself (e.g. "what did I just ask", "summarize this chat").

Classify the question below as exactly one word:
ON_TOPIC - could plausibly be answered from a legal document or refers to the current conversation/documents
OFF_TOPIC - general knowledge, current events, coding, math, creative writing, personal/small talk, or anything else unrelated to reading a legal document

Examples:
Q: What is the termination notice period?
A: ON_TOPIC

Q: What's the capital of France?
A: OFF_TOPIC

Q: Who do you think will win the World Cup?
A: OFF_TOPIC

Q: Can you recommend a good recipe for dinner?
A: OFF_TOPIC

Q: Who are the parties to this agreement?
A: ON_TOPIC

Answer with exactly one word, ON_TOPIC or OFF_TOPIC, nothing else.

Question: {question}
Answer:"""


def is_off_topic_llm(question: str) -> bool:
    """LLM-backed backstop for the keyword gate: classifies whether a question is
    unrelated to legal-document QA, catching phrasings the keyword list can't enumerate."""
    messages = [HumanMessage(content=_OFF_TOPIC_CLASSIFIER_PROMPT.format(question=question))]
    result = _invoke_with_retry(llm.invoke, messages)
    verdict = result.content.strip().upper()
    return verdict.startswith("OFF")

# Instructor-patched raw Groq client — forces structured JSON output for verification
_verifier = instructor.from_groq(
    Groq(api_key=os.getenv("GROQ_API_KEY")),
    mode=instructor.Mode.JSON,
)

# Mirrors RAG_SYSTEM_PROMPT_FILE's override pattern so a candidate FACT prompt (e.g.
# system_prompt_v4.txt) can be paired with a matching candidate analytical prompt
# (e.g. system_prompt_analytical_v2.txt) during A/B testing, without the analytical
# routing in stream_answer() silently staying pinned to the old file.
_ANALYTICAL_PROMPT_FILE = os.getenv("RAG_ANALYTICAL_PROMPT_FILE", "system_prompt_analytical.txt")

_prompt_cache: dict = {}

def load_prompt(filename=None):
    """Loads the system prompt, caching each file by name after first read."""
    filename = filename or os.getenv("RAG_SYSTEM_PROMPT_FILE", "system_prompt_v3.txt")
    if filename in _prompt_cache:
        return _prompt_cache[filename]
    prompt_path = os.path.join("prompts", filename)
    with open(prompt_path, "r", encoding="utf-8") as file:
        content = file.read()
    _prompt_cache[filename] = content
    return content

_REFERENCE_RE = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|he|she|him|her|his|hers|the same|aforementioned|previous|latter|former)\b",
    re.IGNORECASE,
)

def needs_rephrasing(question: str) -> bool:
    """Returns True if the question contains pronoun/reference words that require prior context to resolve."""
    return bool(_REFERENCE_RE.search(question))

def rephrase_question(question: str, history: list) -> str:
    """Rephrases a follow-up question into a standalone retrieval query using conversation history."""
    history_text = "\n".join(
        f"{t['role'].capitalize()}: {t['content']}" for t in history
    )
    prompt_text = (
        "Given the conversation history below and a follow-up question, "
        "rephrase the follow-up into a single standalone question that contains "
        "all the context needed to search a document database. "
        "Do not answer it — only rewrite it.\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone question:"
    )
    result = _invoke_with_retry(llm.invoke, [HumanMessage(content=prompt_text)])
    return result.content.strip()


# Retrieval struggles when a question names an abstract legal-clause category (e.g.
# "non-compete") but the actual clause text uses unrelated concrete wording. This gate
# detects that pattern and routes to reading the whole document instead of relying on
# retrieval to find the right chunk (see database.get_full_document_context).
_CLAUSE_CATEGORY_RE = re.compile(
    r"related to [\"']?[\w\s\-/]+[\"']?\s*(that should be reviewed|clause)", re.IGNORECASE
)
_CLAUSE_CATEGORY_KEYWORDS = {
    "non-compete", "non compete", "change of control", "governing law",
    "cap on liability", "limitation of liability", "uncapped liability",
    "audit rights", "termination for convenience", "indemnification",
    "indemnity", "confidentiality clause", "exclusivity", "assignment clause",
    "warranty", "insurance requirement", "ip ownership",
    "intellectual property ownership", "most favored nation",
    "non-solicitation", "non solicitation", "source code escrow",
    "renewal term", "revenue sharing", "minimum commitment",
}


def needs_full_document_read(question: str) -> bool:
    """Returns True if the question names a legal-clause category by its generic name
    rather than describing the clause's actual content — the pattern that hits the
    retrieval vocabulary gap between question and contract wording (see comment above).
    Only meaningful when scoped to a single document — see the caller."""
    q_lower = question.lower()
    if _CLAUSE_CATEGORY_RE.search(q_lower):
        return True
    return any(kw in q_lower for kw in _CLAUSE_CATEGORY_KEYWORDS)


def stream_answer(question: str, formatted_context: str, history: list = None, intent: QueryIntent = QueryIntent.FACT):
    """Yields answer text chunks for SSE streaming, with optional conversation history.

    Routes to the analytical system prompt (more coverage, structured output) for ANALYTICAL intent.
    Temperature stays at 0 in both modes to preserve legal accuracy.
    """
    prompt_file = _ANALYTICAL_PROMPT_FILE if intent == QueryIntent.ANALYTICAL else None
    system_prompt = load_prompt(prompt_file)
    # Fill {context} directly to avoid template-parsing conflicts with legal text in history
    filled_system = system_prompt.replace("{context}", formatted_context)

    messages = [SystemMessage(content=filled_system)]
    for turn in (history or []):
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=question))

    for attempt in range(_MAX_RETRIES):
        try:
            for chunk in llm.stream(messages):
                yield chunk.content
            return
        except RateLimitError as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_parse_retry_after(exc) * (2 ** attempt))


# ── Structured verification schemas ───────────────────────────────────────────

class _ExtractedClaims(BaseModel):
    claims: list[str] = Field(
        description="Every distinct factual claim explicitly stated in the answer text. "
                    "Do not add, infer, or rephrase anything not present in the answer."
    )


class VerifiedClaim(BaseModel):
    claim_text: str = Field(description="The original claim text")
    is_faithful: bool = Field(description="True if the claim is backed by verbatim context text")
    supporting_quote: str = Field(
        description="Exact verbatim phrase from context that proves this claim. "
                    "Empty string if is_faithful is False."
    )


class VerificationReport(BaseModel):
    claims: list[VerifiedClaim]
    verdict: Literal["PASS", "PARTIAL", "FAIL"]
    faithfulness_score: float = Field(ge=0.0, le=1.0)


class _VerifiedClaimsList(BaseModel):
    claims: list[VerifiedClaim]


def _extract_claims(answer: str) -> list[str]:
    """Call 1 — extract factual claims from the answer only. No context provided."""
    for attempt in range(_MAX_RETRIES):
        try:
            result = _verifier.chat.completions.create(
                model="openai/gpt-oss-20b",
                temperature=0,
                max_tokens=512,
                reasoning_effort="low",
                max_retries=1,
                response_model=_ExtractedClaims,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract every distinct factual claim from the ANSWER below.\n"
                            "Focus on: dates, numbers, named parties, obligations, rights, provisions.\n"
                            "Only extract claims explicitly stated in the ANSWER.\n"
                            "Do not infer, add, or rephrase anything beyond what the ANSWER says.\n"
                            "Each claim must be a single self-contained statement.\n"
                            "Extract each claim exactly once — do not repeat, paraphrase, or duplicate claims.\n"
                            "If the answer contains only one factual assertion, return exactly one claim."
                        ),
                    },
                    {"role": "user", "content": f"ANSWER:\n{answer}"},
                ],
            )
            return result.claims
        except RateLimitError as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_parse_retry_after(exc) * (2 ** attempt))
    return []


def _verify_claims(claims: list[str], context: str) -> list[VerifiedClaim]:
    """Call 2 — verify each claim against context. Verdict computed in Python, not by the LLM."""
    claims_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    for attempt in range(_MAX_RETRIES):
        try:
            result = _verifier.chat.completions.create(
                model="openai/gpt-oss-20b",
                temperature=0,
                max_tokens=1024,
                reasoning_effort="low",
                max_retries=1,
                response_model=_VerifiedClaimsList,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "For each claim in CLAIMS, find the exact verbatim phrase in CONTEXT that proves it.\n"
                            "- If found: is_faithful=true, copy the exact phrase into supporting_quote.\n"
                            "- If not found: is_faithful=false, supporting_quote=''.\n"
                            "Return one entry per claim, in the same order as the input list."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"CLAIMS:\n{claims_text}\n\nCONTEXT:\n{context}",
                    },
                ],
            )
            return result.claims
        except RateLimitError as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_parse_retry_after(exc) * (2 ** attempt))
    return []


def _build_report(verified: list[VerifiedClaim]) -> VerificationReport:
    """Compute verdict and score from verified claims — no LLM involved."""
    seen_claims: set[str] = set()
    deduped: list[VerifiedClaim] = []
    for vc in verified:
        key = vc.claim_text.strip().lower()
        if key not in seen_claims:
            seen_claims.add(key)
            deduped.append(vc)
    verified = deduped
    total = len(verified)
    if total == 0:
        return VerificationReport(claims=[], verdict="PASS", faithfulness_score=1.0)
    verified_count = sum(1 for c in verified if c.is_faithful)
    score = verified_count / total
    if verified_count == total:
        verdict = "PASS"
    elif verified_count == 0:
        verdict = "FAIL"
    else:
        verdict = "PARTIAL"
    return VerificationReport(claims=verified, verdict=verdict, faithfulness_score=score)


def verify_answer_claims(question: str, context: str, generated_answer: str) -> VerificationReport:
    """
    Two-call faithfulness audit:
      1. Extract claims from the answer only (no context — prevents context bleed).
      2. Verify each extracted claim against the retrieved context.
    Verdict and score are computed in Python from the verified claims list.
    Falls back to legacy PASS/FAIL if either call fails.
    """
    import json as _json
    truncated_context = context[:6000]
    try:
        raw_claims = _extract_claims(generated_answer)
        seen: set[str] = set()
        claims: list[str] = []
        for c in raw_claims:
            key = c.strip().lower()
            if key not in seen:
                seen.add(key)
                claims.append(c)
        if not claims:
            report = VerificationReport(claims=[], verdict="PASS", faithfulness_score=1.0)
        else:
            verified = _verify_claims(claims, truncated_context)
            report = _build_report(verified)
    except Exception as exc:
        try:
            is_pass = _check_faithfulness_legacy(question, context, generated_answer)
        except Exception:
            is_pass = False
        report = VerificationReport(
            claims=[],
            verdict="PASS" if is_pass else "FAIL",
            faithfulness_score=1.0 if is_pass else 0.0,
        )

    os.makedirs("logs", exist_ok=True)
    with open("logs/verification_debug.log", "a", encoding="utf-8") as _f:
        _f.write(_json.dumps({
            "answer": generated_answer[:120],
            "verdict": report.verdict,
            "score": report.faithfulness_score,
            "total_claims": len(report.claims),
            "claims": [{"claim": c.claim_text, "faithful": c.is_faithful} for c in report.claims],
        }) + "\n")
    return report


def _check_faithfulness_legacy(question: str, context: str, generated_answer: str) -> bool:
    """Simple PASS/FAIL auditor — fallback only, used inside verify_answer_claims."""
    eval_prompt_text = (
        "You are a strict grading auditor.\n"
        "Compare the GENERATED ANSWER to the PROVIDED CONTEXT.\n"
        "If the GENERATED ANSWER contains ANY facts, numbers, or claims NOT explicitly "
        "stated in the PROVIDED CONTEXT, output the word FAIL.\n"
        "If the GENERATED ANSWER is entirely supported, output the word PASS.\n\n"
        "PROVIDED CONTEXT:\n{context}\n\n"
        "GENERATED ANSWER:\n{answer}\n\n"
        "Output ONLY 'PASS' or 'FAIL'. Do not explain."
    )
    eval_prompt = ChatPromptTemplate.from_template(eval_prompt_text)
    chain = eval_prompt | llm | StrOutputParser()
    result = _invoke_with_retry(chain.invoke, {"context": context, "answer": generated_answer})
    return "PASS" in result.strip().upper()
