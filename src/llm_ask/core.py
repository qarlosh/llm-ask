"""The provider-neutral part: request, result, errors, backend choice and the call log."""

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

BACKEND_ENV = "LLM_ASK_BACKEND"
DEFAULT_BACKEND = "claude_cli"


class LLMError(RuntimeError):
    """The LLM could not give an answer. Catch this to fall back to deterministic code."""


class LLMUnavailable(LLMError):
    """No LLM to call: CLI not installed or not logged in, no API key, missing SDK."""


class LLMFailed(LLMError):
    """The call was made but gave no usable answer: error, timeout, refusal or invalid output."""


@dataclass(frozen=True)
class Request:
    parts: list[str]
    system: str
    model: str
    schema: dict | None
    effort: str | None
    timeout: float
    setting_sources: str
    prompt_caching: bool


@dataclass(frozen=True)
class Reply:
    """What a backend returns: the answer and what it cost."""

    output: Any
    model: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Result:
    output: Any
    """The parsed object when a schema was given, else the answer text."""
    model: str
    prompt_version: str
    backend: str
    input_tokens: int
    output_tokens: int
    seconds: float


def _call_backend(name: str, request: Request) -> Reply:
    if name == "claude_cli":
        from . import claude_cli

        return claude_cli.call(request)
    if name == "anthropic_api":
        from . import anthropic_api

        return anthropic_api.call(request)
    raise ValueError(f"{BACKEND_ENV}={name!r}: expected 'claude_cli' or 'anthropic_api'")


_log_lock = threading.Lock()


def _log(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def ask(
    prompt: str | list[str],
    *,
    system: str,
    model: str,
    step: str,
    prompt_version: str,
    schema: dict | None = None,
    effort: str | None = None,
    log_path: str | Path | None = None,
    timeout: float = 600,
    setting_sources: str = "",
    prompt_caching: bool = True,
    validate: Callable[[Any], None] | None = None,
) -> Result:
    """Sends one request to the LLM chosen by LLM_ASK_BACKEND and returns its answer.

    `prompt` may be a list of parts, ordered from the most stable to the most variable; the API
    backend caches up to the second-to-last one. With `schema` (a JSON Schema) the answer is the
    parsed object; without it, the text. `step` and `prompt_version` go to the log and the result.
    `setting_sources` only applies to `claude -p`; `prompt_caching=False` turns caching off,
    which is cheaper for prompts that are read once.

    `validate` checks the output beyond the schema (ranges, ids that exist): it raises ValueError
    to reject the answer, which then fails like any other bad answer.

    Raises LLMUnavailable or LLMFailed (both LLMError). Every call, failed or not, is appended
    to `log_path` as one JSON line when given.
    """
    backend = os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND
    request = Request(
        parts=[prompt] if isinstance(prompt, str) else list(prompt),
        system=system,
        model=model,
        schema=schema,
        effort=effort,
        timeout=timeout,
        setting_sources=setting_sources,
        prompt_caching=prompt_caching,
    )
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "step": step,
        "backend": backend,
        "model": model,
        "prompt_version": prompt_version,
    }
    start = time.monotonic()
    try:
        reply = _call_backend(backend, request)
        if validate is not None:
            try:
                validate(reply.output)
            except ValueError as e:
                raise LLMFailed(f"output rejected: {e}") from e
    except LLMError as e:
        if log_path:
            entry.update(seconds=round(time.monotonic() - start, 2), error=f"{type(e).__name__}: {e}"[:500])
            _log(Path(log_path), entry)
        raise
    seconds = round(time.monotonic() - start, 2)
    if log_path:
        entry.update(model=reply.model, input_tokens=reply.input_tokens, output_tokens=reply.output_tokens, seconds=seconds)
        _log(Path(log_path), entry)
    return Result(
        output=reply.output,
        model=reply.model,
        prompt_version=prompt_version,
        backend=backend,
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
        seconds=seconds,
    )
