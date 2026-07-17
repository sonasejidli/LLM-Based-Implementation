"""
Turning a raw LLM text response into a validated SupportResponse.

Models asked for "JSON only" will still sometimes:
  - wrap it in ```json ... ``` fences
  - add a sentence before or after it ("Sure, here's the JSON: {...}")
  - leave a trailing comma before a closing brace/bracket
  - get cut off mid-object if max_tokens is too low
  - return something that isn't JSON at all

This module does NOT assume clean output. It tries a sequence of
increasingly forgiving strategies to recover a JSON object, and only
raises once all of them fail. Schema validation (pydantic) is a
separate, final step — extraction succeeding just means "this parses
as JSON", not "this is the object we asked for".
"""
import json
import re

from pydantic import ValidationError

from .schema import SupportResponse


class JSONExtractionError(Exception):
    """Raised when no valid, schema-conforming JSON could be recovered."""


def _strip_code_fences(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            return candidate
    return text


def _find_balanced_object(text: str, start: int) -> str | None:
    """Walk forward from the first '{' and return the substring up to its
    matching '}', respecting braces that appear inside quoted strings."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # never balanced -> likely truncated


def extract_json_object(text: str) -> str:
    """Return a string that json.loads() can parse, or raise JSONExtractionError."""
    if not text or not text.strip():
        raise JSONExtractionError("Empty response from model")

    text = _strip_code_fences(text.strip())

    # Fast path: already clean JSON
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise JSONExtractionError(
            f"No '{{' found in model response; got: {text[:200]!r}"
        )

    # Try to pull a balanced {...} block out of surrounding prose
    candidate = _find_balanced_object(text, start)
    if candidate is not None:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass  # fall through to cleanup attempt below

    # Last resort: first '{' to last '}' in the whole text, with light
    # cleanup for the most common garbling (trailing commas).
    end = text.rfind("}")
    if end == -1 or end <= start:
        raise JSONExtractionError(
            f"JSON object looks truncated (no closing brace found); "
            f"got: {text[:200]!r}"
        )
    candidate = text[start : end + 1]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)  # drop trailing commas

    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError as e:
        raise JSONExtractionError(
            f"Found JSON-like text but couldn't parse it ({e}); "
            f"got: {text[:200]!r}"
        )


def parse_support_response(raw_text: str) -> SupportResponse:
    """Extract JSON from raw_text and validate it against SupportResponse.
    Raises JSONExtractionError on any failure (bad JSON or schema mismatch),
    so callers have one exception type to catch and retry on."""
    json_str = extract_json_object(raw_text)
    data = json.loads(json_str)

    try:
        return SupportResponse.model_validate(data)
    except ValidationError as e:
        raise JSONExtractionError(f"JSON parsed but failed schema validation: {e}")
