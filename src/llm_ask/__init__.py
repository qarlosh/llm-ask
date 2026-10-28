"""One provider-neutral `ask()` for LLM calls, with a call log, typed errors and a pytest guard."""

from .core import LLMError, LLMFailed, LLMUnavailable, Result, ask
from .untrusted import untrusted

__all__ = ["LLMError", "LLMFailed", "LLMUnavailable", "Result", "ask", "untrusted"]
