"""Rotating-key Gemini client used as ClauseIQ's production generation model.

Google AI Studio's free tier caps gemini-2.5-flash at 20 requests/day PER PROJECT
(confirmed the hard way in the 2026-09-21 swap attempt — see the
clauseiq-generation-model-swap memory). This app has no billing enabled and is
prototype-scale traffic only, so instead of paying for a higher tier we pool several
free-tier keys/projects behind one client: on a quota/rate-limit error it moves on to
the next usable key rather than failing the request (see RotatingGeminiChat). This only helps because
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
import re
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


AI_QUOTA_EXHAUSTED_MESSAGE = (
    "ClauseIQ has reached its AI usage limit for now. Please try again later — "
    "the limit resets daily."
)


class GeminiUnavailableError(RuntimeError):
    """No configured key can serve a request right now — every key is out of quota,
    cooling down, or permanently unusable. Callers should tell the user to try later
    rather than show a generic failure."""


# How long a key sits out after a 429. Google's retryDelay (~20-60s) is only
# meaningful for per-minute limits; for the per-day free-tier cap it just invites a
# retry that fails again, so a daily-quota key is benched for an hour before one
# cheap re-probe (the quota resets at midnight Pacific time).
_DAILY_QUOTA_COOLDOWN_S = 3600
_DEFAULT_COOLDOWN_S = 60


def _cooldown_seconds(exc: ClientError) -> float:
    text = str(exc)
    if "PerDay" in text:
        return _DAILY_QUOTA_COOLDOWN_S
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", text) or re.search(r"'retryDelay': '(\d+)s'", text)
    return float(m.group(1)) if m else _DEFAULT_COOLDOWN_S


class RotatingGeminiChat:
    """LangChain-chat-model-shaped wrapper around google-genai that rotates across
    multiple API keys on quota/rate-limit/capacity errors.

    Each request tries every currently-usable key at most once, without sleeping
    between keys — waiting only ever helped the old single-key setup, and with several
    dead or exhausted keys it stacked up to 30+s of sleep before a generic failure.
      - 429 (quota): the key is benched for its cooldown, then tried again.
      - 403/404 (e.g. "model no longer available to new users" on newer projects):
        the key is permanently unusable for this model and skipped from then on.
      - 5xx: transient — move on to the next key after a short pause.
      - any other 4xx (a bad request, not a bad key) is raised immediately.
    If no key succeeds, GeminiUnavailableError is raised (unless the last failure was
    a transient 5xx, which is re-raised as is).
    """

    def __init__(self, model: str, api_keys: list[str]) -> None:
        if not api_keys:
            raise RuntimeError(
                "GEMINI_API_KEY is not set (comma-separated for multiple keys/projects)."
            )
        self._model = model
        self._clients = [genai.Client(api_key=k) for k in api_keys]
        self._index = 0  # last key that worked — tried first next time
        self._dead: set[int] = set()
        self._cool_until: dict[int, float] = {}

    def _usable_keys(self) -> list[int]:
        now = time.time()
        n = len(self._clients)
        order = [(self._index + i) % n for i in range(n)]
        return [i for i in order if i not in self._dead and self._cool_until.get(i, 0) <= now]

    def _handle_error(self, idx: int, exc: Exception) -> None:
        """Records what a failed key's error means for its future use; re-raises
        errors that aren't the key's fault."""
        code = getattr(exc, "code", None)
        if isinstance(exc, ServerError):
            logger.warning("Gemini key #%d server error (%s); trying next key", idx, str(exc)[:120])
            time.sleep(1)
        elif code == 429:
            wait = _cooldown_seconds(exc)
            self._cool_until[idx] = time.time() + wait
            logger.warning("Gemini key #%d out of quota; benched for %ds", idx, wait)
        elif code in (403, 404):
            self._dead.add(idx)
            logger.error("Gemini key #%d unusable for %s, skipping it from now on: %s",
                         idx, self._model, str(exc)[:200])
        else:
            raise exc

    def _unavailable(self, last_exc: Exception | None) -> Exception:
        if isinstance(last_exc, ServerError):
            return last_exc
        dead, cooling = len(self._dead), len(self._cool_until)
        logger.error("No usable Gemini key (%d configured, %d unusable, %d benched for quota)",
                     len(self._clients), dead, cooling)
        return GeminiUnavailableError("All Gemini API keys are out of quota or unavailable.")

    def invoke(self, messages: list) -> _Chunk:
        system_instruction, contents = _to_gemini_contents(messages)
        cfg = genai_types.GenerateContentConfig(system_instruction=system_instruction, temperature=0)
        last_exc: Exception | None = None
        for idx in self._usable_keys():
            try:
                resp = self._clients[idx].models.generate_content(model=self._model, contents=contents, config=cfg)
                self._index = idx
                return _Chunk(resp.text or "")
            except (ClientError, ServerError) as exc:
                last_exc = exc
                self._handle_error(idx, exc)
        raise self._unavailable(last_exc)

    def stream(self, messages: list):
        """Yields _Chunk objects. Moves to another key only while no chunk of this
        attempt has been yielded yet, so a mid-stream failure never re-emits duplicate
        output to an SSE client — it just propagates."""
        system_instruction, contents = _to_gemini_contents(messages)
        cfg = genai_types.GenerateContentConfig(system_instruction=system_instruction, temperature=0)
        last_exc: Exception | None = None
        for idx in self._usable_keys():
            yielded_any = False
            try:
                for chunk in self._clients[idx].models.generate_content_stream(
                    model=self._model, contents=contents, config=cfg
                ):
                    yielded_any = True
                    yield _Chunk(chunk.text or "")
                self._index = idx
                return
            except (ClientError, ServerError) as exc:
                last_exc = exc
                if yielded_any:
                    raise
                self._handle_error(idx, exc)
        raise self._unavailable(last_exc)
