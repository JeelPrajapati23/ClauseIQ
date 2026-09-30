import os
import re
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.pdf_parser import extract_pages_from_pdf

# A heading line: "ARTICLE 8" / "Section 8.4 Audits." / "8.4 Records; Audits. <body…>" /
# "8.3.1 <body…>" / "1. Duties of the Administrator" / "1.76 “Net Sales” means…".
# Numbered headings in contracts usually run straight into their body text on the
# same line, so the title is only taken when it looks like one (short Title Case
# ending in a period or at end of line, or a quoted defined term) — otherwise just
# the number is used. Spacing is [^\S\n] rather than [ \t]: many PDFs extract with
# non-breaking spaces (\xa0) throughout, which [ \t] silently never matched.
_HEADING_RE = re.compile(
    r"^[^\S\n]*(?:"
    r"(?P<kw>ARTICLE|Article|SECTION|Section|SCHEDULE|Schedule|EXHIBIT|Exhibit|ANNEX|Annex|APPENDIX|Appendix)"
    r"[^\S\n]+(?P<kwnum>\d+(?:\.\d+)*|[IVXLC]+|[A-Z])\b\.?(?P<kwrest>[^\n]*)"
    r"|(?P<num>\d{1,2}(?:\.\d{1,3})+)\.?(?:[^\S\n]+(?P<rest>[^\n]*))?$"
    r"|(?P<num1>\d{1,2})[.)][^\S\n]+(?P<rest1>[^\n]*)"
    r")",
    re.MULTILINE,
)
_TITLE_RE = re.compile(r"\s*([A-Z][^.\n]{0,70}?)(?:\.(?:\s|$)|\s*$)")
_DEFINED_TERM_RE = re.compile(r"\s*([“\"][^”\"\n]{1,60}[”\"])")
# A "heading" whose previous line ends like this is really a sentence that wrapped
# after a cross-reference ("...pursuant to Section" / "11.4 IS INTENDED...").
_WRAPPED_REF_RE = re.compile(r"(?:\b(?:sections?|articles?|and|or|of|to|in|under|with|per|this)|,)\s*$", re.I)
# A bare number with no title is only trusted when the previous line ended a sentence
# or block — otherwise it's likely a wrapped line starting with a figure ("2.5 times").
_BLOCK_END_RE = re.compile(r"[.:;)\]”\"*]\s*$")
_TITLE_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"}


def _title(text: str) -> str:
    """Leading heading title from text, or "" if it reads like a sentence instead."""
    text = text or ""
    term = _DEFINED_TERM_RE.match(text)
    if term:
        return term.group(1)
    m = _TITLE_RE.match(text)
    if not m:
        return ""
    title = m.group(1).strip()
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", title)
    if not words or len(words) > 10:
        return ""
    if any(w[0].islower() and w.lower() not in _TITLE_SMALL_WORDS for w in words):
        return ""
    return title


def _heading_label(m: re.Match, prev_line: str) -> str:
    if m.group("num1"):
        # Single-level "1." is also how plain numbered lists look — require a real title.
        title = _title(m.group("rest1"))
        return f"{m.group('num1')}. {title}" if title else ""
    if m.group("num"):
        title = _title(m.group("rest"))
        if not title and prev_line.strip() and not _BLOCK_END_RE.search(prev_line):
            return ""
        return f"{m.group('num')} {title}".strip()
    kw = m.group("kw").capitalize()
    rest = m.group("kwrest").strip()
    title = _title(rest) if rest else ""
    if rest and not title:
        return ""  # "ARTICLE 11. Aimmune shall provide..." — a wrapped sentence, not a heading
    return f"{kw} {m.group('kwnum')}" + (f" {title}" if title else "")


def find_headings(text: str) -> list[tuple[int, str]]:
    """Every heading in text as (start offset, label), in order."""
    found = []
    for m in _HEADING_RE.finditer(text):
        prev_line = text[:m.start()].rstrip("\n").rsplit("\n", 1)[-1]
        if _WRAPPED_REF_RE.search(prev_line):
            continue
        label = _heading_label(m, prev_line)
        if label:
            found.append((m.start(), " ".join(label.split())[:100]))
    return found


def heading_before(text: str, offset: int) -> str:
    """Label of the last heading starting at or before offset, or ""."""
    label = ""
    for start, lbl in find_headings(text):
        if start > offset:
            break
        label = lbl
    return label


# Parents: broad legal sections (no overlap — sections must be distinct)
# Children: tight fragments for dense semantic retrieval
PARENT_CHUNK_SIZE = 2000
CHILD_CHUNK_SIZE = 350
CHILD_CHUNK_OVERLAP = 50

_parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE,
    chunk_overlap=0,
    separators=["\n\n", "\n", " "],
)
_child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE,
    chunk_overlap=CHILD_CHUNK_OVERLAP,
)


def process_pdf(file_path: str):
    """
    Two-tier parent-child chunking for legal PDFs.

    Parents (2000 chars, no overlap): broad legal sections split at paragraph
    boundaries. Stored as payload on each child — sent to the LLM at generation time.

    Children (350 chars, 50 overlap): tight, high-signal fragments. Embedded as
    vectors and indexed in Qdrant — used for semantic + BM25 retrieval.

    Every child carries:
      parent_context  : full text of its parent section
      parent_id       : UUID shared by all siblings from the same parent (used for
                        deduplication at retrieval time)
      section         : heading in effect at the child's start (carried over from
                        earlier chunks/pages when the child has none of its own)
      section_start   : heading in effect at the start of the parent chunk
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    source_file = os.path.basename(file_path)
    pages = extract_pages_from_pdf(file_path)  # [(page_num, text), ...]

    page_docs = [
        Document(
            page_content=text,
            metadata={
                "source": file_path,
                "source_file": source_file,
                "page": page_num,
            },
        )
        for page_num, text in pages
    ]

    child_chunks = []
    # Heading in effect so far — carried across chunk and page boundaries, since a
    # clause often continues well past the chunk that holds its heading.
    carried = ""
    for parent in _parent_splitter.split_documents(page_docs):
        parent_id = str(uuid.uuid4())
        text = parent.page_content
        section_start = carried
        children = _child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_context"] = text
            child.metadata["parent_id"] = parent_id
            child.metadata["section_start"] = section_start
            offset = max(text.find(child.page_content[:80]), 0)
            section = heading_before(text, offset) or section_start
            if section:
                child.metadata["section"] = section
        child_chunks.extend(children)
        carried = heading_before(text, len(text)) or carried

    return child_chunks
