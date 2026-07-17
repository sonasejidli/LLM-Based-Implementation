"""
Ties the prompt, the model call, and the parser together, with a
retry loop for when the model doesn't return clean JSON on the first try.

Design choice: on retry, we don't just resend the same prompt and hope
for a different roll of the dice. We append an explicit correction
instruction, since the model is a fresh call each time and doesn't
remember it just failed.
"""
from . import config
from .llm_client import call_model
from .parser import JSONExtractionError, parse_support_response
from .schema import SupportResponse

SYSTEM_PROMPT = """You are a customer support triage assistant.

Given a customer message, respond with ONLY a single JSON object — no \
prose, no markdown code fences, no explanation before or after it.

The JSON object must have exactly these fields:
{
  "category": one of "billing" | "technical" | "account" | "general" | "complaint",
  "sentiment": one of "positive" | "neutral" | "negative" | "angry",
  "priority": one of "low" | "medium" | "high" | "urgent",
  "suggested_response": a short, empathetic reply to send the customer (string),
  "confidence": a number between 0 and 1 for how confident you are in this triage
}

Return raw JSON only."""

RETRY_SUFFIX = """

Your previous response could not be parsed as valid JSON matching the \
schema above. This time return ONLY the raw JSON object — no markdown \
fences, no commentary, no leading or trailing text of any kind."""


def triage_message(customer_message: str) -> SupportResponse:
    """Call the model and return a validated SupportResponse, retrying
    up to config.MAX_RETRIES times if the output isn't valid, schema-
    conforming JSON."""
    last_error: Exception | None = None
    last_raw: str | None = None

    for attempt in range(1, config.MAX_RETRIES + 1):
        system = SYSTEM_PROMPT if attempt == 1 else SYSTEM_PROMPT + RETRY_SUFFIX
        raw = call_model(system=system, user=customer_message)
        last_raw = raw
        try:
            return parse_support_response(raw)
        except JSONExtractionError as e:
            last_error = e
            continue

    raise JSONExtractionError(
        f"Failed to get valid structured output after {config.MAX_RETRIES} "
        f"attempt(s). Last error: {last_error}\nLast raw response: {last_raw!r}"
    )
