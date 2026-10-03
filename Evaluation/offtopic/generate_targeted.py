"""Step 1b-ii: targeted generation for the three categories the first trained classifier
confidently blocked on the test set (see results/train_report.json, first run):

  mixed     ON-10  a document question combined with an unrelated request
  meta      ON-3   questions about the conversation itself
  app_usage ON-9   questions about what ClauseIQ can do / how to use it

plus OFF_TOPIC contrast sets, so the classifier learns the actual boundary rather than
"addressed to 'you' = ON" or "two requests in one = ON":

  assistant_chitchat OFF-6  personal small talk aimed at the assistant (name, feelings, jokes)
  multi_offtopic     OFF    two unrelated non-legal requests in one message

Written to data/raw/generated_targeted.jsonl with source "generated_targeted". These rows
only ever go to train/val (build_splits.py never puts them in the test pool), because the
categories were chosen by looking at test errors — reusing the test set to judge a fix it
inspired would overstate the improvement.

The first generation run copied its prompt's example phrasings verbatim (one follow-up
appeared 15 times), so these prompts describe what to write without quotable examples,
and each call gets a different persona and topic for variety.

Usage: python Evaluation/offtopic/generate_targeted.py   (from the project venv)
"""
import hashlib
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _groq import chat_json, n_keys  # noqa: E402

OUT_PATH = Path(__file__).parent / "data" / "raw" / "generated_targeted.jsonl"
MAX_CHARS = 2000
CALLS = {"mixed": 20, "meta": 20, "app_usage": 20, "assistant_chitchat": 10, "multi_offtopic": 10}
PRIOR = {"mixed": "ON_TOPIC", "meta": "ON_TOPIC", "app_usage": "ON_TOPIC",
         "assistant_chitchat": "OFF_TOPIC", "multi_offtopic": "OFF_TOPIC"}

random.seed(23)

PERSONAS = [
    "a busy lawyer who types tersely", "a non-native English speaker who makes small grammar mistakes",
    "a small business owner writing casually", "a university student", "a very polite, formal person",
    "someone typing all lowercase with no punctuation", "someone texting on a phone with abbreviations",
    "an anxious first-time tenant", "an HR manager", "a startup founder in a hurry",
]
DOC_TYPES = [
    "NDA", "residential lease", "employment contract", "SaaS agreement", "privacy policy",
    "loan agreement", "franchise agreement", "insurance policy", "supply agreement", "terms of service",
]
OFF_THEMES = [
    "cooking", "travel plans", "sports scores", "a coding bug", "the weather", "a birthday gift",
    "fitness", "a movie recommendation", "maths homework", "calendar reminders", "music", "pets",
]

COMMON = """You are generating training data for a classifier that guards ClauseIQ, a chat app where users upload legal documents and ask questions about them. Write as {persona}. Vary length, wording and structure across the items — no two items may share an opening phrase, and do not use stock phrasings that a template would produce.

"""

PROMPTS = {
    "mixed": COMMON + """Write 15 messages, each combining (a) a genuine question or request about the user's uploaded {doctype} with (b) an unrelated request about {theme}. Put the unrelated part first in about half of them. Join the parts in different ways (one sentence, two sentences, a list, an aside in brackets, an afterthought). Make the unrelated part sometimes short and sometimes the longer half.
Return JSON: {{"items": ["...", ...]}}""",
    "meta": COMMON + """Write 15 messages a user sends about the conversation itself while discussing their {doctype} — not about the document's content. Cover: recalling what they or the assistant said earlier, recaps and summaries of the chat so far, asking to repeat or rephrase the last answer, asking what an earlier answer meant, asking which question they asked first, asking to continue or go back to an earlier point, asking whether something was already covered. Some should be very short.
Return JSON: {{"items": ["...", ...]}}""",
    "app_usage": COMMON + """Write 15 messages a user sends asking about the ClauseIQ assistant itself — what it can do and how to use it — while working with a {doctype}. Cover: what kinds of questions they can ask, which file types or document kinds it handles, comparing documents, how it finds answers and cites sources, how reliable it is, limits on length or number of files, whether uploaded documents are kept private, how to start over, what topics it knows about. Include some very short, vague ones of the kind people type into a help box.
Return JSON: {{"items": ["...", ...]}}""",
    "assistant_chitchat": COMMON + """Write 15 OFF-TOPIC messages: personal small talk aimed at the assistant that has nothing to do with documents or with using the app — its name, age, feelings, opinions, favourite things, whether it is human, requests to tell a joke, sing, or chat for fun. Loosely touch on {theme} in a few of them.
Return JSON: {{"items": ["...", ...]}}""",
    "multi_offtopic": COMMON + """Write 15 OFF-TOPIC messages that each combine two unrelated everyday requests, neither of which is about a legal document, law, or the app — e.g. mixing {theme} with some other everyday topic. Join the parts in different ways (one sentence, two sentences, a list, an afterthought). Never mention contracts, agreements, policies, clauses, terms or anything legal.
Return JSON: {{"items": ["...", ...]}}""",
}


def _seed_id(kind: str, n: int) -> str:
    return f"generated_targeted-{kind}-{hashlib.sha1(f'{kind}|{n}'.encode()).hexdigest()[:12]}"


def run(kind: str, n: int) -> list[dict]:
    prompt = PROMPTS[kind].format(persona=PERSONAS[n % len(PERSONAS)], doctype=random.choice(DOC_TYPES),
                                  theme=random.choice(OFF_THEMES))
    out = chat_json(prompt, temperature=1.0)
    sid = _seed_id(kind, n)
    rows = []
    for text in out.get("items", []):
        text = " ".join(str(text).split())[:MAX_CHARS]
        if text:
            rows.append({"text": text, "source": "generated_targeted", "prior": PRIOR[kind],
                         "seed_id": sid, "style": kind})
    return rows


def main():
    done = set()
    if OUT_PATH.exists():
        done = {json.loads(line)["seed_id"] for line in OUT_PATH.open(encoding="utf-8")}
    jobs = [(k, n) for k, count in CALLS.items() for n in range(count) if _seed_id(k, n) not in done]
    print(f"{len(done)} seeds done, {len(jobs)} to generate")
    failed = 0
    with ThreadPoolExecutor(max_workers=n_keys()) as pool, OUT_PATH.open("a", encoding="utf-8") as f:
        futures = [pool.submit(run, k, n) for k, n in jobs]
        for fut in as_completed(futures):
            try:
                rows = fut.result()
            except Exception as exc:  # rerun resumes failed seeds
                failed += 1
                print(f"  seed failed: {exc!r}")
                continue
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
    print(f"done ({failed} failed)")


if __name__ == "__main__":
    main()
