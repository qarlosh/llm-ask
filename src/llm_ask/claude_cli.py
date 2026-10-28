"""Backend `claude_cli`: `claude -p` from Claude Code, billed to the caller's own Claude subscription.

The call is stripped to the minimum: our own system prompt, no tools, no MCP servers, no slash
commands or skills, no settings files, no saved session, and an empty temporary working
directory so no project CLAUDE.md is loaded. The whole input travels in the one user message.
"""

import json
import os
import subprocess
import tempfile

from .core import LLMFailed, LLMUnavailable, Reply, Request

# Words in an error that mean the CLI has no valid login, so the LLM is not there at all
AUTH_HINTS = ("not logged in", "/login", "authenticate", "oauth", "401", "unauthorized", "credential")


def command(request: Request) -> list[str]:
    cmd = [
        "claude", "-p",
        "--model", request.model,
        "--system-prompt", request.system,
        "--tools", "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--setting-sources", request.setting_sources,
        "--no-session-persistence",
        "--output-format", "json",
    ]  # fmt: skip
    if request.effort:
        cmd += ["--effort", request.effort]
    if request.schema is not None:
        cmd += ["--json-schema", json.dumps(request.schema)]
    return cmd


def _run(cmd: list[str], stdin: str, cwd: str, env: dict, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, input=stdin, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def _error(message: str) -> LLMFailed | LLMUnavailable:
    if any(hint in message.lower() for hint in AUTH_HINTS):
        return LLMUnavailable(f"claude -p is not logged in: {message}")
    return LLMFailed(f"claude -p: {message}")


def call(request: Request) -> Reply:
    env = dict(os.environ)
    if not request.prompt_caching:
        env["DISABLE_PROMPT_CACHING"] = "1"
    with tempfile.TemporaryDirectory(prefix="llm-ask-") as cwd:
        try:
            done = _run(command(request), "\n\n".join(request.parts), cwd, env, request.timeout)
        except FileNotFoundError as e:
            raise LLMUnavailable("claude CLI not found; install Claude Code and log in") from e
        except subprocess.TimeoutExpired as e:
            raise LLMFailed(f"claude -p timed out after {request.timeout:g} s") from e

    try:
        result = json.loads(done.stdout)
    except json.JSONDecodeError:
        raise _error((done.stderr or done.stdout or f"exit code {done.returncode}").strip()[-500:]) from None
    if result.get("is_error") or done.returncode != 0:
        raise _error(str(result.get("result") or result.get("subtype") or done.stderr).strip()[-500:])

    if request.schema is None:
        output = result.get("result")
    else:
        output = result.get("structured_output")
        if output is None:
            raise LLMFailed(f"claude -p returned no structured output: {str(result.get('result'))[-500:]}")

    usage = result.get("usage") or {}
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_write = usage.get("cache_creation_input_tokens") or 0
    # modelUsage lists every model the CLI used; the one that wrote the most is the one that answered
    by_model = result.get("modelUsage") or {}
    model = max(by_model, key=lambda m: by_model[m].get("outputTokens") or 0, default=request.model)
    return Reply(
        output=output,
        model=model,
        input_tokens=(usage.get("input_tokens") or 0) + cache_read + cache_write,
        output_tokens=usage.get("output_tokens") or 0,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        cost_usd=result.get("total_cost_usd"),
        turns=result.get("num_turns") or 1,
    )
