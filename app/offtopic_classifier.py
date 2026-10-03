"""Trained off-topic classifier: logistic regression over the question's Arctic query
embedding, exported by Evaluation/offtopic/train_classifier.py to
app/models/offtopic_classifier.json and scored here with numpy alone.

It returns a calibrated p(off-topic) and a three-way decision: "block" (p >= t_high),
"allow" (p <= t_low), or "defer" to the Gemini classifier in between. Thresholds were set
on validation data so the classifier on its own blocks <=1% of on-topic questions — see
Evaluation/offtopic/LABELLING_GUIDE.md for what counts as on-topic and results/ for the
test-set numbers.

This is a cost/quality gate, NOT a security boundary: anything phrased with enough legal
vocabulary scores on-topic (by design — a pasted clause or a mixed legal/off-topic request
is on-topic per the guide), so it can't stop a determined user from steering the answer
model. User isolation is enforced by the user_id filters in app/database.py, not here.

Every failure — missing/mismatched model file, embedding API error — returns None so the
caller falls back to the Gemini classifier rather than failing the request.
"""
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

from app.database import EMBEDDING_MODEL, embeddings

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "models" / "offtopic_classifier.json"

Decision = Literal["block", "allow", "defer"]


class OffTopicClassifier:
    def __init__(self, spec: dict):
        self.weights = np.asarray(spec["weights"], dtype=np.float64)
        self.bias = float(spec["bias"])
        self.platt_a = float(spec["platt_a"])
        self.platt_c = float(spec["platt_c"])
        self.t_low = float(spec["t_low"])
        self.t_high = float(spec["t_high"])

    def p_off(self, vector: list[float]) -> float:
        logit = float(np.dot(self.weights, vector)) + self.bias
        return float(1.0 / (1.0 + np.exp(-(self.platt_a * logit + self.platt_c))))

    def decide(self, p_off: float) -> Decision:
        if p_off >= self.t_high:
            return "block"
        if p_off <= self.t_low:
            return "allow"
        return "defer"


@lru_cache(maxsize=1)
def load_classifier() -> OffTopicClassifier | None:
    try:
        spec = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Off-topic classifier not loaded (%s); using the Gemini classifier only", MODEL_PATH,
                       exc_info=True)
        return None
    # Weights are only meaningful in the embedding space they were trained in — a
    # changed embedding model would silently produce garbage scores, so refuse instead.
    if spec.get("embedding_model") != EMBEDDING_MODEL or len(spec.get("weights", [])) != spec.get("dim"):
        logger.warning("Off-topic classifier was trained on %r, app embeds with %r; disabled until retrained",
                       spec.get("embedding_model"), EMBEDDING_MODEL)
        return None
    return OffTopicClassifier(spec)


def classify_off_topic(question: str) -> tuple[float, Decision] | None:
    """(p_off, decision) for the question, or None if the classifier can't run."""
    clf = load_classifier()
    if clf is None:
        return None
    try:
        vector = embeddings.embed_query(question)
    except Exception:
        logger.warning("Off-topic classifier: embedding failed; deferring to the Gemini classifier", exc_info=True)
        return None
    if len(vector) != len(clf.weights):
        logger.warning("Off-topic classifier: got a %d-dim embedding, expected %d", len(vector), len(clf.weights))
        return None
    p = clf.p_off(vector)
    return p, clf.decide(p)
