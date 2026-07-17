"""
Configuration loader.

The API key is NEVER hardcoded. It's read from the environment, which is
populated from a local .env file (via python-dotenv) that is git-ignored.
Only .env.example (with a placeholder) is committed to the repo.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the working directory if present; no-op in prod
# where real env vars are injected by the host (e.g. Docker, CI secrets, etc.)


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


def get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ConfigError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key, e.g.:\n"
            "  cp .env.example .env\n"
            "  echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env"
        )
    return key


# Model + retry behavior are configurable but never secret, so plain
# env vars with sensible defaults are fine here.
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "500"))
