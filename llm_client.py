"""
Thin wrapper around the Anthropic API. The key point: the client is
constructed with the key from app.config (which reads the environment),
never a literal string.
"""
from openai import OpenAI

from . import config

_client = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.get_api_key())
    return _client


def call_model(system: str, user: str, max_tokens: int | None = None) -> str:
    client = get_client()
    response = client.messages.create(
        model=config.MODEL,
        max_tokens=max_tokens or config.MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text_blocks = [block.text for block in response.content if block.type == "text"]
    return "\n".join(text_blocks)
