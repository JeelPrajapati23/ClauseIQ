"""Row-identity helpers shared by the off-topic dataset scripts (dependency-free, so the
training script can import them without pulling in app.database)."""
import hashlib
import re

NEAR_DUP_PREFIX = 100


def text_key(text: str) -> str:
    """Exact identity — the embedding cache key."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def near_dup_key(text: str) -> str:
    """Same normalised first NEAR_DUP_PREFIX chars = near-duplicate (e.g. WildChat's many
    copies of one prompt template, or a generated example repeated across seeds)."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()[:NEAR_DUP_PREFIX]
