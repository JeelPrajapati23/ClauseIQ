"""
Recover which ingested PDF each golden_qa_set.json question was written
against, by matching reference_contexts text back to the PDFs in
Evaluation/ingestion_pdfs/.

golden_qa_set.json never persisted a source_file field, but reference_contexts
holds text extracted from the original CUAD contract (via the same
app.pdf_parser.extract_pages_from_pdf used for ingestion), so a normalized
substring match against our own extracted PDF text is reliable.

Usage:
    python Evaluation/build_golden_doc_map.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pdf_parser import extract_pages_from_pdf

PDF_DIR = Path(__file__).resolve().parent / "ingestion_pdfs"
GOLDEN_SET_FILE = Path(__file__).resolve().parent / "golden_qa_set.json"
OUTPUT_FILE = Path(__file__).resolve().parent / "golden_doc_map.json"

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()


def load_pdf_texts() -> dict[str, str]:
    texts = {}
    # Case-insensitive: Linux (WSL2's /mnt/c included) treats *.pdf and *.PDF as
    # different globs even on a case-preserving Windows-backed filesystem.
    pdf_files = sorted(p for p in PDF_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    for idx, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{idx}/{len(pdf_files)}] Extracting: {pdf_path.name}")
        pages = extract_pages_from_pdf(str(pdf_path))
        texts[pdf_path.name] = normalize("\n".join(t for _, t in pages))
    return texts


def find_matches(snippet: str, pdf_texts: dict[str, str]) -> list[str]:
    """Try shrinking snippet lengths until exactly one (or a few) PDFs match."""
    for length in (400, 250, 150, 80):
        probe = snippet[:length]
        if len(probe) < 40:
            continue
        matches = [name for name, text in pdf_texts.items() if probe in text]
        if matches:
            return matches
    return []


def find_matches_by_paragraph(raw_context: str, pdf_texts: dict[str, str]) -> list[str]:
    """Fallback for contiguous-snippet misses caused by whitespace/reflow drift:
    vote per-paragraph and return the PDF(s) with the most paragraph hits."""
    votes: dict[str, int] = {}
    for para in raw_context.split("\n\n"):
        probe = normalize(para)[:200]
        if len(probe) < 30:
            continue
        for name, text in pdf_texts.items():
            if probe in text:
                votes[name] = votes.get(name, 0) + 1
    if not votes:
        return []
    best = max(votes.values())
    return [name for name, v in votes.items() if v == best]


_WORD_RE = re.compile(r"[a-z]{4,}")


def find_matches_fuzzy(snippet: str, pdf_texts: dict[str, str]) -> list[str]:
    """Last-resort fallback for HAND_WRITTEN snippets that were retyped/reformatted
    rather than copy-pasted, so no exact substring (contiguous or per-paragraph)
    appears anywhere in the extracted PDF text.

    Scores each PDF by IDF-weighted overlap of the snippet's words against that PDF's
    vocabulary: words that appear in most/all of the corpus (generic contract
    boilerplate — "agreement", "shall", "party") carry no signal, while words rare
    across the corpus (party names, defined terms, deal-specific language) are
    strongly diagnostic of the source document even when the surrounding wording
    was reworded. Requires both a minimum score and a clear margin over the
    runner-up, so an ambiguous case is left unmatched rather than silently guessed.
    """
    snippet_words = set(_WORD_RE.findall(snippet.lower()))
    if not snippet_words:
        return []

    pdf_word_sets = {name: set(_WORD_RE.findall(text)) for name, text in pdf_texts.items()}

    doc_freq: dict[str, int] = {}
    for w in snippet_words:
        doc_freq[w] = sum(1 for words in pdf_word_sets.values() if w in words)

    n_docs = len(pdf_texts)
    common_cutoff = max(2, n_docs // 2)
    scores = {
        name: sum(
            1.0 / doc_freq[w]
            for w in snippet_words & words
            if doc_freq[w] < common_cutoff
        )
        for name, words in pdf_word_sets.items()
    }

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked or ranked[0][1] == 0:
        return []
    best_name, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score < 3 or best_score < second_score * 1.5:
        return []
    return [best_name]


def main() -> None:
    with open(GOLDEN_SET_FILE, "r", encoding="utf-8") as f:
        golden = json.load(f)

    pdf_texts = load_pdf_texts()
    print(f"\nLoaded {len(pdf_texts)} PDF texts. Matching {len(golden)} golden rows...\n")

    mapping: dict[str, dict] = {}
    unmatched = 0
    ambiguous = 0

    for source_row, row in enumerate(golden):
        contexts = row.get("reference_contexts") or []
        if not contexts:
            mapping[str(source_row)] = {"source_file": None, "reason": "no_reference_contexts"}
            unmatched += 1
            continue

        snippet = normalize(contexts[0])
        matches = find_matches(snippet, pdf_texts)
        match_reason = "matched"
        if not matches:
            matches = find_matches_by_paragraph(contexts[0], pdf_texts)
        if not matches:
            matches = find_matches_fuzzy(snippet, pdf_texts)
            match_reason = "matched_fuzzy"

        if len(matches) == 1:
            mapping[str(source_row)] = {"source_file": matches[0], "reason": match_reason}
        elif len(matches) > 1:
            mapping[str(source_row)] = {"source_file": matches[0], "reason": f"ambiguous:{matches}"}
            ambiguous += 1
        else:
            mapping[str(source_row)] = {"source_file": None, "reason": "no_match"}
            unmatched += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    matched = len(golden) - unmatched
    print(f"Matched: {matched}/{len(golden)}  (ambiguous: {ambiguous}, unmatched: {unmatched})")
    print(f"Saved mapping to {OUTPUT_FILE}")

    if unmatched:
        print("\nUnmatched rows:")
        for k, v in mapping.items():
            if v["source_file"] is None:
                print(f"  source_row={k}: {v['reason']}")


if __name__ == "__main__":
    main()
