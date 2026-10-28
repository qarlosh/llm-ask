# llm-ask

One `ask()` for the LLM calls inside small Python apps. The app says what it wants (system prompt,
input, JSON Schema, model) and gets back the parsed answer, with the model and prompt version that
produced it. Which LLM answers is chosen by an environment variable, not by the app's code.

- **Two backends for now**, both Claude:
  - `claude_cli` (default): `claude -p` from Claude Code, billed to your own Claude subscription.
  - `anthropic_api`: the Anthropic API with `ANTHROPIC_API_KEY`.
- **Minimal `claude -p` startup:** your own system prompt, no tools, no MCP servers, no slash
  commands or skills, no settings files (configurable), no saved session, structured output with
  `--json-schema`, and an empty temporary working directory.
- **Call log:** one JSON line per call (step, backend, model, prompt version, tokens, seconds, error).
- **Typed errors** so the app can fall back to its deterministic path.
- **`untrusted()`** wraps third-party text in tags and tells the model it is data, not instructions.
- **pytest guard:** real LLM calls are blocked in every test, and a `fake_llm` fixture gives canned
  answers.

## Which backend to use

`claude_cli` is for **your own personal use** with **your own** Claude subscription: tools you run
for yourself on your machine. Do not use it to serve other people; for that, use `anthropic_api`
with an API key.

| `LLM_ASK_BACKEND` | Needs | Model names |
|---|---|---|
| `claude_cli` (default) | Claude Code installed and logged in (`claude` on the `PATH`) | aliases (`haiku`, `sonnet`, `opus`) or full ids |
| `anthropic_api` | `llm-ask[api]` and `ANTHROPIC_API_KEY` | full ids (`claude-haiku-5-5`) |

## Install

From git, pinned to a version tag (there is no PyPI release):

```bash
uv add git+https://github.com/qarlosh/llm-ask@v0.1.0
```

With the API backend:

```bash
uv add "llm-ask[api] @ git+https://github.com/qarlosh/llm-ask@v0.1.0"
```

Requires Python 3.12+.

## Example

```python
from llm_ask import LLMError, ask, untrusted

SYSTEM = "You classify product reviews. Answer only with the requested structured output."
SCHEMA = {
    "type": "object",
    "properties": {"sentiment": {"enum": ["positive", "negative", "mixed"]}},
    "required": ["sentiment"],
}


def classify(review: str) -> str:
    try:
        result = ask(
            ["Classify the sentiment of this review.", untrusted(review, tag="review")],
            system=SYSTEM,
            model="haiku",
            schema=SCHEMA,
            step="classify",
            prompt_version="classify-v1",
            log_path="data/llm.jsonl",
        )
    except LLMError:
        return "unknown"  # the deterministic fallback; tell the user the LLM was not used
    # result.model and result.prompt_version are worth storing next to the answer
    return result.output["sentiment"]
```

## `ask()`

```python
ask(prompt, *, system, model, step, prompt_version, schema=None, effort=None,
    log_path=None, timeout=600, setting_sources="", prompt_caching=True) -> Result
```

- `prompt`: a string, or a list of parts ordered from the most stable to the most variable. All of
  the input goes here; the model reads nothing else. With `anthropic_api`, the parts up to the
  second-to-last are cached.
- `schema`: a JSON Schema. With it, `Result.output` is the parsed object; without it, the text.
- `effort`: `low`, `medium`, `high`, `xhigh` or `max`, when the model supports it.
- `log_path`: append one JSON line per call to this file, failed calls included.
- `setting_sources`: `claude_cli` only. Empty by default; pass `"project"` if the call needs
  something from the project's settings.
- `prompt_caching=False`: turns prompt caching off (`DISABLE_PROMPT_CACHING=1` for `claude -p`).
  Cheaper when each input is read only once, since writing to the cache costs more than reading
  without it.

`Result` has `output`, `model` (the one that answered), `prompt_version`, `backend`,
`input_tokens` (cached ones included), `output_tokens` and `seconds`.

`ask()` is synchronous. From async code, use `await asyncio.to_thread(ask, ...)`.

### Errors

Both are subclasses of `LLMError`:

- `LLMUnavailable`: there is no LLM to call. The `claude` CLI is missing or not logged in, the API
  key is missing or rejected, the API is unreachable, or the `anthropic` package is not installed.
- `LLMFailed`: the call was made but gave no usable answer: an error, a timeout, a refusal, a cut
  answer or output that does not parse.

An unknown `LLM_ASK_BACKEND` raises `ValueError`.

### `untrusted(text, tag="untrusted_data")`

Puts third-party text (a web page, an email, a transcript) between `<tag>` and `</tag>`, right after
a sentence that says it is data and not instructions. Any `<tag>` or `</tag>` inside the text is
escaped so it cannot close the block. This lowers the risk of prompt injection; it does not remove
it. Still check the output beyond the schema: ranges, ids that exist.

## Tests

The package registers a pytest plugin. Wherever `llm-ask` is installed, every test runs with real
LLM calls blocked: reaching `claude -p` or the Anthropic API raises `RealLLMCallBlocked`. It is not an
`LLMError`, so the app's fallback does not hide it.

To test code that calls `ask()`, use the `fake_llm` fixture:

```python
from llm_ask import LLMUnavailable


def test_classify(fake_llm):
    fake_llm.respond({"sentiment": "positive"}, LLMUnavailable("down"))
    assert classify("Great") == "positive"
    assert classify("Great") == "unknown"
    assert "Great" in fake_llm.requests[0].parts[1]
```

`respond()` queues the outputs of the next calls in order; an exception is raised instead of
returned. `requests` holds what each call sent.

## Development

```bash
uv sync
uv run pytest
```

## License

MIT
