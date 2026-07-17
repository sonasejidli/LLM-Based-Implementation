"""
The contract the LLM's JSON output must satisfy.

Using pydantic here means "valid JSON" isn't good enough on its own —
the object also has to have the right fields, the right types, and
values inside allowed ranges/enums. That's the second line of defense
after JSON parsing itself.
"""
from pydantic import BaseModel, Field
from typing import Literal


class SupportResponse(BaseModel):
    category: Literal["billing", "technical", "account", "general", "complaint"]
    sentiment: Literal["positive", "neutral", "negative", "angry"]
    priority: Literal["low", "medium", "high", "urgent"]
    suggested_response: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
