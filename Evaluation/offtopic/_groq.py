"""Minimal key-rotating Groq JSON client shared by the off-topic dataset scripts.

Standalone rather than importing Evaluation/evaluate_rag_offline.py's RotatingChatGroq,
since that module imports ragas (-> pyarrow), which Smart App Control blocks on Windows.
Keys come from RAGAS_GROQ_API_KEYS (comma-separated), else GROQ_API_KEY. Each key is
capped at 8000 tokens/minute, so a 429 benches that key until Groq's reset time and the
call moves to the next key; it only sleeps when every key is benched.
"""
import json
import os
import re
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "openai/gpt-oss-120b"
# Per-minute limits clear within ~1 min; staying limited this long means a daily cap.
GIVE_UP_AFTER_S = 15 * 60


def _load_keys() -> list[str]:
    keys = [k.strip() for k in os.getenv("RAGAS_GROQ_API_KEYS", "").split(",") if k.strip()]
    if not keys and os.getenv("GROQ_API_KEY"):
        keys = [os.getenv("GROQ_API_KEY")]
    if not keys:
        raise RuntimeError("Set RAGAS_GROQ_API_KEYS or GROQ_API_KEY in .env")
    return keys


_clients = [Groq(api_key=k, max_retries=0) for k in _load_keys()]
_benched_until = [0.0] * len(_clients)
_next = 0
_lock = threading.Lock()  # scripts call chat_json from a thread pool, one worker per key


def _retry_after(exc: RateLimitError) -> float:
    header = exc.response.headers.get("retry-after") if exc.response is not None else None
    try:
        return float(header) + 1
    except (TypeError, ValueError):
        return 20.0


def n_keys() -> int:
    return len(_clients)


def chat_json(prompt: str, max_tokens: int = 4000, temperature: float = 0.0) -> dict | list:
    """One chat call returning parsed JSON, rotating keys on rate limits."""
    global _next
    deadline = time.time() + GIVE_UP_AFTER_S
    while time.time() < deadline:
        with _lock:
            now = time.time()
            free = [i for i in range(len(_clients)) if _benched_until[i] <= now]
            if free:
                i = min(free, key=lambda k: (k - _next) % len(_clients))
                _next = (i + 1) % len(_clients)
        if not free:
            time.sleep(max(0.5, min(_benched_until) - now))
            continue
        try:
            resp = _clients[i].chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                # gpt-oss spends part of max_tokens on hidden reasoning; keep it short
                # (same setting as app/generator.py's verifier and the Ragas judge).
                reasoning_effort="low",
                response_format={"type": "json_object"},
            )
        except RateLimitError as exc:
            _benched_until[i] = time.time() + _retry_after(exc)
            continue
        text = resp.choices[0].message.content or ""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                return json.loads(match.group(0))
            raise
    raise RuntimeError(f"Groq: every key stayed rate-limited for {GIVE_UP_AFTER_S}s")
