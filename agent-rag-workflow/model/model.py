"""LLM configuration and instance management."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


@lru_cache(maxsize=1)
def get_llm(model: str = None, temperature: float = 0.0) -> ChatOpenAI:
    """
    Get configured LLM instance with caching.

    Args:
        model: Model name (defaults to env var OPENAI_MODEL or "gpt-4o-mini")
        temperature: Temperature setting (default: 0.0)

    Returns:
        Configured ChatOpenAI instance
    """
    return ChatOpenAI(
        temperature=temperature, model=model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
    )


# Default LLM instance (for backward compatibility)
llm = get_llm()
