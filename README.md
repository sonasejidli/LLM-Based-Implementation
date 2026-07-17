# Support Responder

A small CLI app that triages a customer support message using the Anthropic
API: it classifies category/sentiment/priority and drafts a suggested reply,
returned as structured JSON.

Built to demonstrate two things explicitly:

1. **No hardcoded API key.** The key is read from the environment via
   `python-dotenv` (`app/config.py`), loaded from a local `.env` file that is
   git-ignored. Only `.env.example` (a placeholder) is committed.
2. **The model will not always return clean JSON — so don't assume it will.**
   `app/parser.py` and `app/responder.py` handle that explicitly (details below).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
python -m app.cli "I was charged twice for my subscription this month and I need a refund."
```

Output:

```json
{
  "category": "billing",
  "sentiment": "negative",
  "priority": "high",
  "suggested_response": "I'm sorry for the double charge...",
  "confidence": 0.9
}
```

## Test

```bash
python -m pytest tests/ -v
```

18 tests, no API key required — they exercise the parser and retry logic
directly with canned/mocked model output, not live API calls.

## The structured-output problem, and how it's handled

Even when a prompt says "return ONLY JSON," real model output sometimes
looks like:

- `` ```json\n{...}\n``` `` — wrapped in a markdown code fence
- `Sure, here's the result: {...} Let me know if you need anything else!` — JSON buried in prose
- `{"a": 1, "b": 2, }` — a trailing comma that breaks strict JSON parsing
- `{"a": 1, "b": 2` — truncated mid-object (e.g. `max_tokens` cut it off)
- `I'm not sure how to help with that.` — no JSON at all

`extract_json_object()` in `app/parser.py` tries, in order:
1. Parse as-is.
2. Strip a markdown code fence and try again.
3. Scan for the first `{`, walk forward tracking brace depth (ignoring
   braces inside quoted strings) to find the matching `}`, and parse that
   balanced substring — this recovers JSON embedded in prose.
4. As a last resort, take everything from the first `{` to the last `}` and
   strip trailing commas before `}`/`]`, then try once more.

If every strategy fails, it raises `JSONExtractionError` with the offending
text included — it never silently returns partial or guessed data.

Parsing success isn't the same as correctness, so `parse_support_response()`
also validates the result against a `pydantic` schema (`app/schema.py`):
right fields, right types, enum values constrained to the allowed set,
`confidence` constrained to `[0, 1]`. A JSON object that parses fine but
has `"category": "not_a_real_category"` is still rejected.

`app/responder.py` wraps all of this in a retry loop (`LLM_MAX_RETRIES`,
default 3). On retry, it doesn't just resend the same prompt — it appends
an explicit "your last response didn't parse, return raw JSON only"
instruction, since each API call is stateless and the model has no memory
of having just failed. If all retries are exhausted, the caller gets a
clear error with the last raw response attached for debugging, rather than
a crash on a `None` or a silently wrong triage.

## Project layout

```
app/
  config.py     # env var loading, API key check
  schema.py     # pydantic contract for the LLM's output
  llm_client.py # Anthropic API call wrapper
  parser.py     # JSON extraction + schema validation (the core of this exercise)
  responder.py  # prompt + retry loop
  cli.py        # command-line entry point
tests/
  test_parser.py     # edge cases: fences, prose, trailing commas, truncation, bad enums
  test_responder.py  # retry loop, mocked LLM calls
```
