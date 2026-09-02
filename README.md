# AI Registration Assistant — DAS008940 (Task AI-SS-001)

Conversational AI assistant that guides students through the Free Online AI & Data Science
internship registration. No database — registration data is stored as JSON.

## What's inside

| Path | Description |
| --- | --- |
| `src/lib/assistant.ts` | NLP engine: preprocessing, intent classification, entity extraction, validation, dialog state machine, sentiment |
| `src/routes/index.tsx` | Web chat interface with a live NLP inspector, slot tracker and `registration.json` preview |
| `python/registration_assistant.py` | NLTK reference implementation of the same logic (CLI chatbot, writes `registrations.json`) |

## How the NLP works

1. **Preprocessing** — lowercase, strip punctuation, tokenize, remove stop words, lemmatize.
2. **Intent classification** — bag-of-words overlap between the message tokens and each intent's
   pattern set, normalised by `sqrt(#patterns)`; below a 0.3 threshold the intent is `unknown`
   and a fallback response is returned.
3. **Entity extraction** — regex patterns for name, email, phone, plus keyword lookups for field
   of study and experience level.
4. **Validation** — per-slot rules (full name, RFC-ish email, 10-digit phone) with corrective
   re-prompts instead of blind acceptance.
5. **Dialog management** — slot-filling state machine over `name → email → phone → field →
   experience`, with FAQ/help/status/restart intents handled at any point and the pending
   question restated afterwards.
6. **Confirmation** — a registration record (`REG-XXXXX`, timestamp, task and student code) is
   emitted as JSON.

## Run the web app

```bash
bun install
bun run dev
```

## Run the Python chatbot

```bash
pip install nltk
python python/registration_assistant.py
```

## Features

Core: greeting, information collection, intent recognition, entity extraction, validation,
registration confirmation.
Bonus: FAQ handling, sentiment analysis, live analytics/inspector panel, web interface.

Task link: https://www.freeinternships.in/
