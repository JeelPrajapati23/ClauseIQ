"""Rotating-key Gemini client used as ClauseIQ's production generation model.

Google AI Studio's free tier caps gemini-2.5-flash at 20 requests/day PER PROJECT
(confirmed the hard way in the 2026-09-21 swap attempt — see the
clauseiq-generation-model-swap memory). This app has no billing enabled and is
prototype-scale traffic only, so instead of paying for a higher tier we pool several
free-tier keys/projects behind one client: on a quota/rate-limit error it rotates to
the next key with backoff rather than failing the request. This only helps because
traffic here is low enough that N keys x 20 req/day is enough headroom — it is not a
real fix for production-scale traffic (see Evaluation/gemini_rotation.py, which uses
the same rotation strategy for offline eval runs).

Exposes invoke()/stream() shaped like a LangChain chat model's (.content on the
result/each chunk) so it drops into app/generator.py and app/compare.py wherever
ChatGroq's `llm` was used directly, without changing call sites.
"""

from __future__ import annotations

import logging
import os
import time

from google import genai
from google.genai import types as genai_types
from google.genai.errors import ClientError, ServerError
from langchain_core.messages import AIMessage, SystemMessage

logger = logging.getLogger(__name__)


def load_gemini_keys(env_var: str = "GEMINI_API_KEY") -> list[str]:
    """Parses a comma-separated env var into a list of API keys."""
    raw = os.getenv(env_var, "")
    return [k.strip() for k in raw.split(",") if k.strip()]


class _Chunk:
    """Minimal stand-in for a LangChain AIMessageChunk — callers only read .content."""

    __slots__ = ("content",)

    def __init__(self, content: str) -> None:
        self.content = content


def _to_gemini_contents(messages: list) -> tuple[str | None, list[dict]]:
    """Splits LangChain-style messages into a Gemini system_instruction + contents list."""
    system_instruction = None
    contents = []
    for m in messages:
        if isinstance(m, SystemMessage):
            system_instruction = m.content
        elif isinstance(m, AIMessage):
            contents.append({"role": "model", "parts": [{"text": m.content}]})
        else:  # HumanMessage or anything else with .content
            contents.append({"role": "user", "parts": [{"text": m.content}]})
    return system_instruction, contents


class RotatingGeminiChat:
    """LangChain-chat-model-shaped wrapper around google-genai that rotates across
    multiple API keys on quota/rate-limit/capacity errors."""

    def __init__(self, model: str, api_keys: list[str], max_attempts: int | None = None) -> None:
        if not api_keys:
            raise RuntimeError(
                "GEMINI_API_KEY is not set (comma-separated for multiple keys/projects)."
            )
        self._model = model
        self._clients = [genai.Client(api_key=k) for k in api_keys]
        self._index = 0
        self._max_attempts = max_attempts or max(len(api_keys) * 2, 4)

    def _rotate(self, reason: str) -> None:
        next_index = (self._index + 1) % len(self._clients)
        logger.warning(
            "Gemini key #%d exhausted/unavailable (%s); rotating to key #%d",
            self._index, reason, next_index,
        )
        self._index = next_index

    def invoke(self, messages: list) -> _Chunk:
        system_instruction, contents = _to_gemini_contents(messages)
        cfg = genai_types.GenerateContentConfig(system_instruction=system_instruction, temperature=0)
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            client = self._clients[self._index]
            try:
                resp = client.models.generate_content(model=self._model, contents=contents, config=cfg)
                return _Chunk(resp.text or "")
            except (ClientError, ServerError) as exc:
                last_exc = exc
                self._rotate(f"{type(exc).__name__}: {str(exc)[:120]}")
                time.sleep(min(30, 2 * (2 ** attempt)))
        raise last_exc  # type: ignore[misc]

    def stream(self, messages: list):
        """Yields _Chunk objects. Rotates+retries only while no chunk of this attempt
        has been yielded yet, so a mid-stream failure never re-emits duplicate output
        to an SSE client — it just propagates."""
        system_instruction, contents = _to_gemini_contents(messages)
        cfg = genai_types.GenerateContentConfig(system_instruction=system_instruction, temperature=0)
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            client = self._clients[self._index]
            yielded_any = False
            try:
                for chunk in client.models.generate_content_stream(
                    model=self._model, contents=contents, config=cfg
                ):
                    yielded_any = True
                    yield _Chunk(chunk.text or "")
                return
            except (ClientError, ServerError) as exc:
                last_exc = exc
                if yielded_any:
                    raise
                self._rotate(f"{type(exc).__name__}: {str(exc)[:120]}")
                time.sleep(min(30, 2 * (2 ** attempt)))
        raise last_exc  # type: ignore[misc]
