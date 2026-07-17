import pytest

from app.parser import JSONExtractionError, extract_json_object, parse_support_response

VALID_OBJ = (
    '{"category": "billing", "sentiment": "negative", "priority": "high", '
    '"suggested_response": "We are sorry for the trouble, let us fix your invoice.", '
    '"confidence": 0.87}'
)


# ---- extract_json_object: things that SHOULD succeed ----

def test_clean_json_passthrough():
    assert extract_json_object(VALID_OBJ) == VALID_OBJ


def test_json_wrapped_in_markdown_fence():
    raw = f"```json\n{VALID_OBJ}\n```"
    result = extract_json_object(raw)
    assert result.strip().startswith("{")


def test_json_wrapped_in_plain_fence_no_language_tag():
    raw = f"```\n{VALID_OBJ}\n```"
    result = extract_json_object(raw)
    assert result.strip().startswith("{")


def test_json_with_prose_before_and_after():
    raw = f"Sure, here's the triage result:\n\n{VALID_OBJ}\n\nLet me know if you need anything else!"
    result = extract_json_object(raw)
    assert result.strip().startswith("{")
    assert result.strip().endswith("}")


def test_json_with_trailing_comma():
    garbled = VALID_OBJ[:-1] + ", }"  # inject trailing comma before close
    result = extract_json_object(garbled)
    import json
    json.loads(result)  # should not raise


def test_json_with_braces_inside_string_values():
    raw = (
        '{"category": "technical", "sentiment": "neutral", "priority": "low", '
        '"suggested_response": "Try restarting the app: {steps here}.", '
        '"confidence": 0.5}'
    )
    result = extract_json_object(raw)
    import json
    parsed = json.loads(result)
    assert "steps here" in parsed["suggested_response"]


# ---- extract_json_object: things that SHOULD fail loudly ----

def test_empty_response_raises():
    with pytest.raises(JSONExtractionError):
        extract_json_object("")


def test_whitespace_only_response_raises():
    with pytest.raises(JSONExtractionError):
        extract_json_object("   \n  ")


def test_no_json_at_all_raises():
    with pytest.raises(JSONExtractionError):
        extract_json_object("I'm not sure how to help with that, sorry!")


def test_truncated_json_raises():
    truncated = VALID_OBJ[: len(VALID_OBJ) // 2]  # cut off mid-object
    with pytest.raises(JSONExtractionError):
        extract_json_object(truncated)


def test_unparseable_garbage_raises():
    with pytest.raises(JSONExtractionError):
        extract_json_object("{category: billing, sentiment: sad clearly not json}")


# ---- parse_support_response: schema validation layer ----

def test_valid_object_parses_to_model():
    result = parse_support_response(VALID_OBJ)
    assert result.category == "billing"
    assert result.priority == "high"
    assert 0.0 <= result.confidence <= 1.0


def test_valid_json_but_wrong_enum_value_raises():
    bad = VALID_OBJ.replace('"billing"', '"not_a_real_category"')
    with pytest.raises(JSONExtractionError):
        parse_support_response(bad)


def test_valid_json_but_missing_field_raises():
    import json
    obj = json.loads(VALID_OBJ)
    del obj["confidence"]
    with pytest.raises(JSONExtractionError):
        parse_support_response(json.dumps(obj))


def test_valid_json_but_confidence_out_of_range_raises():
    bad = VALID_OBJ.replace("0.87", "1.5")
    with pytest.raises(JSONExtractionError):
        parse_support_response(bad)
