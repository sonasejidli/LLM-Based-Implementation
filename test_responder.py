from unittest.mock import patch

import pytest

from app.parser import JSONExtractionError
from app.responder import triage_message

VALID_OBJ = (
    '{"category": "technical", "sentiment": "neutral", "priority": "medium", '
    '"suggested_response": "Let\'s get your app updated.", "confidence": 0.7}'
)


def test_succeeds_on_first_try():
    with patch("app.responder.call_model", return_value=VALID_OBJ) as mock_call:
        result = triage_message("my app keeps crashing")
        assert result.category == "technical"
        assert mock_call.call_count == 1


def test_recovers_after_garbled_first_response():
    # First call returns prose with no JSON, second call returns valid JSON.
    responses = ["Sorry, I don't understand the format you want.", VALID_OBJ]
    with patch("app.responder.call_model", side_effect=responses) as mock_call:
        result = triage_message("my app keeps crashing")
        assert result.category == "technical"
        assert mock_call.call_count == 2
        # second call's system prompt should include the correction suffix
        second_call_kwargs = mock_call.call_args_list[1].kwargs
        assert "could not be parsed" in second_call_kwargs["system"]


def test_raises_after_exhausting_all_retries():
    with patch("app.responder.call_model", return_value="not json at all"):
        with pytest.raises(JSONExtractionError):
            triage_message("my app keeps crashing")
