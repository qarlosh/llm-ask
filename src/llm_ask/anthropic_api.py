"""Backend `anthropic_api`: the Anthropic Messages API with ANTHROPIC_API_KEY.

Needs the optional dependency: `llm-ask[api]`.
"""

import json
import os

from .core import LLMFailed, LLMUnavailable, Reply, Request

MAX_TOKENS = 16000


def params(request: Request) -> dict:
    content = [{"type": "text", "text": text} for text in request.parts]
    # Cache up to the second-to-last part: the last one is what changes between calls
    if request.prompt_caching and len(content) > 1:
        content[-2]["cache_control"] = {"type": "ephemeral"}
    output_config = {}
    if request.effort:
        output_config["effort"] = request.effort
    if request.schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": request.schema}
    p = {
        "model": request.model,
        "max_tokens": MAX_TOKENS,
        "system": request.system,
        "messages": [{"role": "user", "content": content}],
    }
    if output_config:
        p["output_config"] = output_config
    return p


def _create(p: dict, timeout: float):
    import anthropic

    return anthropic.Anthropic().messages.create(**p, timeout=timeout)


def call(request: Request) -> Reply:
    if request.claude_cli.tools or request.claude_cli.mcp_config:
        raise ValueError("tools and MCP servers need the claude_cli backend")
    try:
        import anthropic
    except ImportError as e:
        raise LLMUnavailable("the anthropic package is missing; install llm-ask[api]") from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise LLMUnavailable("ANTHROPIC_API_KEY is not set")

    try:
        message = _create(params(request), request.timeout)
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.APIConnectionError) as e:
        raise LLMUnavailable(f"Anthropic API: {e}") from e
    except anthropic.APIError as e:
        raise LLMFailed(f"Anthropic API: {e}") from e

    if message.stop_reason == "refusal":
        raise LLMFailed("the model refused the request")
    if message.stop_reason == "max_tokens":
        raise LLMFailed(f"the answer was cut at max_tokens ({MAX_TOKENS})")
    text = "".join(block.text for block in message.content if block.type == "text")
    if request.schema is None:
        output = text
    else:
        try:
            output = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMFailed(f"invalid JSON from the model: {text[-500:]}") from e

    u = message.usage
    cache_read = u.cache_read_input_tokens or 0
    cache_write = u.cache_creation_input_tokens or 0
    return Reply(
        output=output,
        model=message.model,
        input_tokens=u.input_tokens + cache_read + cache_write,
        output_tokens=u.output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )
