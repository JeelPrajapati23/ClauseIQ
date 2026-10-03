"""Step 1b of the off-topic classifier dataset: generate questions the public sources
don't cover, into data/raw/generated.jsonl. See LABELLING_GUIDE.md.

Public data has only ~2k natural ON_TOPIC questions, none phrased the way ClauseIQ users
ask about contracts, and all of them short — while WildChat's OFF_TOPIC prompts are long
(median ~430 chars). Left alone, the classifier could learn "long = off-topic" and block
a user pasting a clause. So this generates, per seed:

  clause seeds (real CUAD clauses): formal / casual-with-typos / fragment / follow-up /
      advisory / drafting questions, plus programmatic pasted-text variants of 300-1800
      chars built from the real contract text around the clause (long ON_TOPIC rows)
  doctype seeds: the same styles for document types CUAD lacks (leases, employment,
      privacy policies...) plus conversation-meta, ClauseIQ-usage and mixed requests
  hard-negative seeds: OFF_TOPIC rows that share surface features with ON_TOPIC ones —
      legal-topic writing tasks (OFF-7), pasted non-legal text with a question, polished
      capitalised questions with "?" (CLINC's are all lowercase with no "?")

`prior` is the label each row was generated to have; label_with_llm.py labels every row
independently and flags disagreements for human review. Every row from one seed shares
a `seed_id`, so paraphrases never straddle the train/test split. Resumable: seeds already
in the output file are skipped.

Usage: python Evaluation/offtopic/generate_questions.py   (from the project venv)
"""
import hashlib
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _groq import chat_json, n_keys  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CUAD_PATH = ROOT / "Evaluation" / "CUAD_v1.json"
OUT_PATH = Path(__file__).parent / "data" / "raw" / "generated.jsonl"
MAX_CHARS = 2000

N_CLAUSE_SEEDS = 200
DOCTYPE_CALLS_EACH = 2
N_HARD_NEGATIVE_CALLS = 24

random.seed(7)

DOC_TYPES = [
    "residential lease", "commercial lease", "employment contract", "independent contractor agreement",
    "non-disclosure agreement", "website terms of service", "app privacy policy", "insurance policy",
    "loan agreement", "share purchase agreement", "settlement agreement", "last will and testament",
    "partnership agreement", "software license / SaaS subscription agreement", "court judgment or order",
]

CLAUSE_PROMPT = """You are generating realistic test questions for a legal-document Q&A app. A user has uploaded a contract and is asking about it. Here is one real clause from it.

Contract: {title}
Clause category: {category}
Clause text: "{clause}"

Write questions a real user might type about this clause or this part of the contract. Vary them like real users do — do NOT just restate the category name. Return JSON:
{{"questions": [
  {{"style": "formal", "text": "a complete, well-formed question"}},
  {{"style": "casual", "text": "lowercase, informal, maybe a typo or missing punctuation"}},
  {{"style": "casual", "text": "another informal one, different wording"}},
  {{"style": "fragment", "text": "2-5 words, like a search box query"}},
  {{"style": "followup", "text": "a follow-up that only makes sense mid-conversation, e.g. 'what about if they breach it?' — no contract-specific nouns"}},
  {{"style": "advisory", "text": "asks for risk, negotiation or what-should-I-do advice about this clause"}},
  {{"style": "drafting", "text": "asks to draft/rewrite something grounded in this clause, e.g. an email to the other party or a plain-English version"}},
  {{"style": "paste_question", "text": "a short question someone would type right after pasting this clause, e.g. 'is this normal?' or 'what does this mean for me'"}}
]}}"""

DOCTYPE_PROMPT = """You are generating realistic test questions for a legal-document Q&A app called ClauseIQ. Users upload legal documents and ask questions about them. Assume the user has uploaded a {doctype}.

Write 16 varied questions a real user might type. Mix lengths, tone (formal, casual, typos, all lowercase, no punctuation), and include:
- 6 about specific content of the document (rights, obligations, dates, amounts, termination, liability...)
- 2 short fragments (2-5 words)
- 2 follow-ups that only make sense mid-conversation
- 2 asking for advice or risks ("should I sign this", "what should I push back on")
- 1 about the conversation itself ("summarise what we discussed", "what did I ask earlier")
- 1 about using the app ("can you compare two documents?", "what can you do")
- 1 general legal question related to this kind of document that does not mention the document
- 1 mixed request that combines a question about the document with an unrelated request
Return JSON: {{"questions": [{{"style": "<one of: content, fragment, followup, advisory, meta, app_usage, general_legal, mixed>", "text": "..."}}]}}"""

HARD_NEGATIVE_PROMPT = """You are generating OFF-TOPIC test inputs for a classifier that guards a legal-document Q&A app. The app must refuse anything unrelated to reading or asking about a legal document. Generate inputs that are off-topic but LOOK superficially similar to on-topic ones, so the classifier can't rely on surface features. Theme for variety: {theme}.

Write 12 inputs:
- 3 writing tasks on a legal topic that are not about any specific document (e.g. "Write a 500-word essay on the history of contract law", "Write a blog post about famous lawsuits")
- 3 that paste 2-5 sentences of NON-legal text (news article, recipe, product review, email from a friend, code snippet) followed by a question or instruction about it — make these 300-900 characters long
- 3 polished general-knowledge or personal-assistant questions in normal sentence case (not all caps) ending in "?"
- 3 off-topic requests that use words like "document", "terms", "agreement", "policy" or "clause" in a non-legal sense (e.g. "Summarise this document about photosynthesis", "What are the terms of a geometric series?", "Explain the subordinate clause in this German sentence") — never treaties, statutes or real legal instruments
Return JSON: {{"questions": [{{"style": "<one of: legal_topic_writing, pasted_nonlegal, polished_general, legal_word_nonlegal>", "text": "..."}}]}}"""

HARD_NEGATIVE_THEMES = [
    "sports", "cooking", "travel", "programming", "personal finance", "health and fitness",
    "movies and TV", "science", "history", "school homework", "careers", "technology news",
]


def _seed_id(kind: str, key: str) -> str:
    return f"generated-{kind}-{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def _row(text: str, prior: str, seed_id: str, style: str) -> dict | None:
    text = " ".join(str(text).split())[:MAX_CHARS]
    if not text:
        return None
    return {"text": text, "source": "generated", "prior": prior, "seed_id": seed_id, "style": style}


def clause_seeds() -> list[dict]:
    data = json.loads(CUAD_PATH.read_text(encoding="utf-8"))["data"]
    candidates = []
    for doc in data:
        para = doc["paragraphs"][0]
        for qa in para["qas"]:
            if not qa["answers"]:
                continue
            category = qa["question"].split('related to "')[1].split('"')[0]
            if category in ("Document Name", "Parties"):
                continue  # one-line answers, too thin to seed varied questions
            ans = qa["answers"][0]
            if len(ans["text"]) < 60:
                continue
            candidates.append({"title": doc["title"], "category": category, "clause": ans["text"][:1200],
                               "context": para["context"], "start": ans["answer_start"]})
    # One clause per (contract, category), spread across contracts and categories.
    random.shuffle(candidates)
    seen, seeds = set(), []
    for c in candidates:
        if (c["title"], c["category"]) not in seen:
            seen.add((c["title"], c["category"]))
            seeds.append(c)
    return seeds[:N_CLAUSE_SEEDS]


def pasted_variants(seed: dict, paste_question: str | None, sid: str) -> list[dict]:
    """Long ON_TOPIC rows built from real contract text, so length isn't an OFF_TOPIC tell."""
    ctx = seed["context"]
    length = random.randint(300, 1800)
    begin = max(0, seed["start"] - random.randint(0, 300))
    # Snap to word boundaries so excerpts don't start or end mid-word.
    if begin > 0:
        begin = ctx.find(" ", begin) + 1
    end = ctx.rfind(" ", begin, begin + length)
    excerpt = " ".join(ctx[begin:end if end > begin else begin + length].split())
    rows = [_row(excerpt, "ON_TOPIC", sid, "pasted_bare")]
    if paste_question:
        rows.append(_row(f'"{seed["clause"]}" {paste_question}', "ON_TOPIC", sid, "pasted_clause_question"))
        rows.append(_row(f"{paste_question}\n\n{excerpt}", "ON_TOPIC", sid, "question_then_paste"))
    return [r for r in rows if r]


def run_clause_seed(seed: dict) -> list[dict]:
    sid = _seed_id("clause", f'{seed["title"]}|{seed["category"]}')
    out = chat_json(CLAUSE_PROMPT.format(title=seed["title"], category=seed["category"],
                                         clause=seed["clause"]), temperature=0.9)
    qs = out.get("questions", [])
    rows = [_row(q.get("text", ""), "ON_TOPIC", sid, q.get("style", "")) for q in qs
            if q.get("style") != "paste_question"]
    paste_q = next((q.get("text") for q in qs if q.get("style") == "paste_question"), None)
    return [r for r in rows if r] + pasted_variants(seed, paste_q, sid)


def run_doctype(doctype: str, n: int) -> list[dict]:
    sid = _seed_id("doctype", f"{doctype}|{n}")
    out = chat_json(DOCTYPE_PROMPT.format(doctype=doctype), temperature=0.9)
    # general_legal (ON-6) and mixed (ON-10) are ON_TOPIC by the guide, like the rest.
    return [r for r in (_row(q.get("text", ""), "ON_TOPIC", sid, q.get("style", ""))
                        for q in out.get("questions", [])) if r]


def run_hard_negative(theme: str, n: int) -> list[dict]:
    sid = _seed_id("hardneg", f"{theme}|{n}")
    out = chat_json(HARD_NEGATIVE_PROMPT.format(theme=theme), temperature=0.9)
    return [r for r in (_row(q.get("text", ""), "OFF_TOPIC", sid, q.get("style", ""))
                        for q in out.get("questions", [])) if r]


def main():
    done = set()
    if OUT_PATH.exists():
        done = {json.loads(line)["seed_id"] for line in OUT_PATH.open(encoding="utf-8")}

    jobs = []
    for seed in clause_seeds():
        if _seed_id("clause", f'{seed["title"]}|{seed["category"]}') not in done:
            jobs.append((run_clause_seed, (seed,)))
    for doctype in DOC_TYPES:
        for n in range(DOCTYPE_CALLS_EACH):
            if _seed_id("doctype", f"{doctype}|{n}") not in done:
                jobs.append((run_doctype, (doctype, n)))
    for n in range(N_HARD_NEGATIVE_CALLS):
        theme = HARD_NEGATIVE_THEMES[n % len(HARD_NEGATIVE_THEMES)]
        if _seed_id("hardneg", f"{theme}|{n}") not in done:
            jobs.append((run_hard_negative, (theme, n)))

    print(f"{len(done)} seeds already done, {len(jobs)} to generate on {n_keys()} keys")
    failed = 0
    with ThreadPoolExecutor(max_workers=n_keys()) as pool, OUT_PATH.open("a", encoding="utf-8") as f:
        futures = [pool.submit(fn, *args) for fn, args in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                rows = fut.result()
            except Exception as exc:  # one bad seed shouldn't kill the run; rerun resumes it
                failed += 1
                print(f"  seed failed: {exc!r}")
                continue
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if k % 20 == 0:
                print(f"  {k}/{len(jobs)} seeds")
    print(f"done ({failed} failed — rerun to retry them)")


if __name__ == "__main__":
    main()
