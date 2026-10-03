# Off-topic classifier — labelling guide

Every example in the off-topic dataset (training labels drafted by `gpt-oss-120b`, and
the hand-checked test set) is labelled by these rules. The classifier is only a
pre-retrieval gate: an ON_TOPIC question still goes through retrieval and the answer
prompt's own refusal rules, so a wrong ON_TOPIC label costs one wasted LLM call, while
a wrong OFF_TOPIC label blocks a real user. **The labels are deliberately biased toward
ON_TOPIC.**

## The core test

> Could someone reasonably ask this while working with a legal document (contract,
> lease, NDA, policy, terms of service, court filing…), about that document, its
> subject matter, the law around it, or the conversation itself?

Yes → `ON_TOPIC`. Clearly no → `OFF_TOPIC`. **Unsure after applying the rules below →
`ON_TOPIC` with `unsure: true`.**

The label depends only on the question's meaning, never on its form: length, language,
spelling, capitalisation and punctuation are not signals.

## ON_TOPIC

| Rule | What | Examples |
|---|---|---|
| ON-1 | Anything about a document's content: parties, dates, amounts, obligations, clauses, definitions, rights, termination, liability. **Includes questions about what a company, app or service does or allows — data collection, sharing, retention, refunds, cancellation, account rights, age limits — because those are answered by its privacy policy or terms of service, even when phrased to "you" or "the app"** | "What is the termination notice period?" · "who pays for repairs" · "does this app track my location?" · "how long are my messages kept?" · "can I get a refund if I cancel?" |
| ON-2 | Follow-ups and fragments that only make sense with conversation history | "what about the second one?" · "and termination?" · "explain sec 4" · "ok and the penalty??" · "indemnity" |
| ON-3 | Questions about the conversation or the uploaded documents themselves | "summarise this chat" · "what did I just ask?" · "which documents did I upload?" |
| ON-4 | Pasted document text, with or without a question | "'Licensee shall not sublicense…' — what does this mean?" · a pasted clause on its own |
| ON-5 | General legal concepts and terms — *the grey-zone decision (b)* | "What does indemnify mean?" · "What counts as consideration in contract law?" |
| ON-6 | General legal questions and advice, even without mentioning a document | "Can my landlord keep my deposit?" · "Is a non-compete enforceable in California?" · "Can Hawaii secede from the U.S.?" |
| ON-7 | Analysis, risk, comparison or negotiation advice about a document | "What are the risks for me here?" · "What should I push back on?" · "How do these two NDAs differ?" |
| ON-8 | Writing tasks grounded in a document | "Draft an email to the counterparty about clause 5" · "Rewrite this clause in plain English" |
| ON-9 | Questions about using ClauseIQ itself | "What can you do?" · "How do I compare two contracts?" |
| ON-10 | Mixed requests where any part is ON_TOPIC | "Summarise clause 3 and write a poem about it" |

## OFF_TOPIC

| Rule | What | Examples |
|---|---|---|
| OFF-1 | Programming and technical help | "write a python function to reverse a list" · "explain spark.dynamicAllocation.enabled" |
| OFF-2 | Creative writing not grounded in a document | "write a poem about autumn" · "write a YouTube script about…" |
| OFF-3 | General knowledge, trivia, maths, science, history | "capital of France" · "how many primes are there below 100" · "why do nails rust" |
| OFF-4 | Current events, sport, markets, weather | "how much has the Dow changed today" · "who will win the World Cup?" |
| OFF-5 | Personal-assistant and device tasks, including **commands to act on the user's accounts** (as opposed to questions about what a policy says) and personal finance/tax lookups | "set an alarm for 7" · "how do you say fly in Italian" · "dim my screen" · "transfer $50 to carrie" · "how much do I owe in state taxes" |
| OFF-6 | Small talk and chit-chat with no request | "hi" · "how are you?" · "let's play a game" |
| OFF-7 | Academic or general writing and editing not grounded in a document — **even on a legal topic** | "polish my essay" · "write an essay on the history of contract law" |
| OFF-8 | Personal advice outside law | "recipe for dinner" · "how do I get over a breakup?" |
| OFF-9 | Attempts to override the assistant's instructions | "ignore your previous instructions and…" |

## Tie-breakers

1. ON-10 beats every OFF rule: one ON_TOPIC part makes the whole request ON_TOPIC.
2. ON-5/ON-6 (a legal question) vs OFF-7 (a writing task): if it asks for an **answer**,
   it's ON_TOPIC; if it asks for a **piece of writing** not grounded in a document — an
   essay, article, blog post, analysis or report "on" a legal subject — it's OFF_TOPIC,
   however legal the subject is.
3. Short keyword fragments that name legal or contract concepts ("agreement start date
   termination", "patent suit restriction") are ON-2 — they're search-box queries about
   a document, not off-topic noise.
4. Still unsure → `ON_TOPIC`, `unsure: true`. Unsure rows are kept in training but
   reported separately in evaluation, so we can see how much of the error comes from
   the grey zone versus clear-cut cases.

## Record format (`.jsonl`, one row per example)

```json
{"text": "...", "label": "ON_TOPIC", "rule": "ON-5", "unsure": false, "source": "law_stack_exchange", "seed_id": "lse-71340"}
```

- `rule` — the rule that decided the label, so disagreements can be traced to a rule.
- `source` — originating dataset, or `generated` / `chat_message` / `golden_set`. Used to
  report per-source accuracy, which catches the classifier learning a dataset's style
  instead of its topic.
- `seed_id` — groups an example with its paraphrases; train/test splits are made by
  `seed_id`, never by row, so near-duplicates can't leak across the split.

## Known consequence

Decision (b) is broader than the current Gemini classifier prompt
(`_OFF_TOPIC_CLASSIFIER_PROMPT` in `app/generator.py`), which calls general legal
knowledge OFF_TOPIC. When the trained classifier is wired in, that prompt must be updated
to this guide, since it stays as the fallback for uncertain scores and the two must not
disagree. The baseline comparison scores the current prompt against these labels as-is,
so expect some of its "errors" to be ON-5/ON-6 rows.
