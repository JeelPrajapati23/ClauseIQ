"""Shared rotating-key wrapper for calling Gemini as a generation candidate in this eval
harness. Free-tier Gemini keys have a low per-project daily request cap, so a real eval
run needs several keys' quotas pooled together. Used by both
compare_generation_models.py (fast proxy comparison) and evaluate_rag_offline.py
(RAGAS_GENERATION_BACKEND=gemini, full Ragas run).

Requires `pip install google-genai` — not in requirements.txt, since production
generation stays on Groq.
"""

from __future__ import annotations

import os
import time

from google import genai
from google.genai import types as genai_types
from google.genai.errors import ClientError, ServerError


class RotatingGeminiClient:
    """Wraps multiple free-tier Gemini API keys/projects behind one generate() call: on a
    quota/rate-limit or capacity error, rotates to the next key with exponential backoff
    instead of the whole run stalling once one key's daily cap is hit.
    """

    def __init__(self, model: str, api_keys: list[str], max_attempts: int = 8) -> None:
        if not api_keys:
            raise ValueError("RotatingGeminiClient needs at least one API key.")
        self._model = model
        self._clients = [genai.Client(api_key=k) for k in api_keys]
        self._index = 0
        self._max_attempts = max_attempts

    def _rotate(self, reason: str) -> None:
        next_index = (self._index + 1) % len(self._clients)
        print(f"    gemini key #{self._index} exhausted/unavailable ({reason}); rotating to key #{next_index}")
        self._index = next_index

    def generate(self, system_instruction: str, contents: str) -> str:
        cfg = genai_types.GenerateContentConfig(system_instruction=system_instruction, temperature=0)
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            client = self._clients[self._index]
            try:
                resp = client.models.generate_content(model=self._model, contents=contents, config=cfg)
                return resp.text or ""
            except (ServerError, ClientError) as exc:
                last_exc = exc
                self._rotate(f"{type(exc).__name__}: {str(exc)[:120]}")
                # Covers both a transient capacity error and a daily-quota error on the
                # key being rotated off — exponential backoff handles either.
                wait = min(60, 4 * (2 ** attempt))
                print(f"    retrying in {wait}s (attempt {attempt + 1}/{self._max_attempts})")
                time.sleep(wait)
        raise last_exc  # type: ignore[misc]


def load_gemini_keys(env_var: str = "GEMINI_API_KEY") -> list[str]:
    """Parse a comma-separated env var into a list of API keys."""
    raw = os.getenv(env_var, "")
    return [k.strip() for k in raw.split(",") if k.strip()]
