"""Claim-level citations built from the claim verifier's verbatim supporting quotes.

verify_answer_claims() already asks the verifier for an exact supporting_quote per
claim. Here each quote is located back in the retrieved parent chunks, which pins
the claim to a file, page and nearest section heading. A quote that can't be found
in any parent produces no citation, so a hallucinated quote never becomes a source.
"""
import os
import re
from difflib import SequenceMatcher

from app.loader import find_headings

# Fraction of a quote's characters that must appear (in order) in a parent for the
# fuzzy fallback to accept it — the verifier's "verbatim" quotes drift slightly
# (dropped punctuation, a changed article) often enough that exact-only misses them.
FUZZY_MATCH_MIN = 0.85
# Quotes shorter than this are too generic to locate reliably ("the Company").
MIN_QUOTE_CHARS = 12

_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ",
})


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_TRANSLATE)).strip().lower()


def _locate(quote: str, text: str) -> tuple[float, int]:
    """Returns (match quality 0-1, char offset of the match in normalized text).

    1.0 means an exact substring. Otherwise the best similarity of the quote against
    any same-length window of text — windowed so a short quote can't "match" by
    picking scattered characters out of a whole 2000-char parent.
    """
    q, t = _normalize(quote), _normalize(text)
    pos = t.find(q)
    if pos != -1:
        return 1.0, pos
    # Only score windows aligned to where a real run of the quote appears in text.
    blocks = SequenceMatcher(None, q, t, autojunk=False).get_matching_blocks()
    starts = {max(0, b.b - b.a) for b in blocks if b.size >= 8}
    width = len(q) + len(q) // 10
    best, best_pos = 0.0, -1
    for start in starts:
        r = SequenceMatcher(None, q, t[start:start + width], autojunk=False).ratio()
        if r > best:
            best, best_pos = r, start
    return min(best, 0.999), best_pos


def _section_before(text: str, offset: int) -> str:
    """The last heading that starts before offset (an offset into the normalized text)."""
    heading = ""
    for start, label in find_headings(text):
        if len(_normalize(text[:start])) > offset:
            break
        heading = label
    return heading


def _source_file(doc) -> str:
    return doc.metadata.get("source_file", os.path.basename(doc.metadata.get("source", "Unknown")))


def build_claim_citations(claims: list, parent_docs: list) -> list[dict]:
    """
    Maps each faithful VerifiedClaim to the parent chunk containing its quote.

    Returns one dict per located claim, in answer order:
      {claim, quote, file, page, section, exact}
    Claims marked unfaithful, with a too-short quote, or whose quote isn't found
    in any parent are omitted.
    """
    citations = []
    for claim in claims:
        quote = (claim.supporting_quote or "").strip().strip("\"'“”‘’").strip()
        if not claim.is_faithful or len(quote) < MIN_QUOTE_CHARS:
            continue
        best_score, best_doc, best_offset = 0.0, None, -1
        for doc in parent_docs:
            score, offset = _locate(quote, doc.page_content)
            if score > best_score:
                best_score, best_doc, best_offset = score, doc, offset
            if score == 1.0:
                break
        if best_doc is None or best_score < FUZZY_MATCH_MIN:
            continue
        pages = best_doc.metadata.get("all_pages") or [best_doc.metadata.get("page")]
        pages = [p for p in pages if p is not None]
        citations.append({
            "claim": claim.claim_text,
            "quote": quote,
            "file": _source_file(best_doc),
            # Parents never span a page boundary (see app/loader.py), so this is exact.
            "page": pages[0] if pages else None,
            # No heading above the quote in this chunk → the one carried in from earlier
            # chunks (section_start; documents indexed before it existed have none).
            # Not metadata["section"]: that can name a heading *after* the quote.
            "section": _section_before(best_doc.page_content, best_offset)
                       or best_doc.metadata.get("section_start", ""),
            "exact": best_score == 1.0,
        })
    return citations
