"""One provider-neutral `ask()` for LLM calls, with a call log, typed errors and a pytest guard."""

from .core import LLMError, LLMFailed, LLMUnavailable, Result, ask

__all__ = ["LLMError", "LLMFailed", "LLMUnavailable", "Result", "ask"]
