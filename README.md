# llm-ask

One `ask()` for the LLM calls inside small Python apps. The app says what it wants (system prompt,
input, JSON Schema, model) and gets back the parsed answer, with the model and prompt version that
produced it. Which LLM answers is chosen by an environment variable, not by the app's code.

- **Two backends for now**, both Claude:
  - `claude_cli` (default): `claude -p` from Claude Code, billed to your own Claude subscription.
  - `anthropic_api`: the Anthropic API with `ANTHROPIC_API_KEY`.
- **Minimal `claude -p` startup** by default: your own system prompt, no tools, no MCP servers, no
  slash commands or skills, no settings files, no saved session, structured output with
  `--json-schema`, and an empty temporary working directory. An app can turn tools, MCP servers,
  skills and settings back on with `ClaudeCliOptions`.
- **Call log:** one JSON line per call (step, backend, model, prompt version, tokens with cache
  reads and writes, cost, turns, seconds, error).
- **Typed errors** so the app can fall back to its deterministic path, and a `validate` hook to
  reject answers that fit the schema but make no sense.
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
    log_path=None, timeout=600, prompt_caching=True, validate=None,
    claude_cli=ClaudeCliOptions()) -> Result
```

- `prompt`: a string, or a list of parts ordered from the most stable to the most variable. All of
  the input goes here; the model reads nothing else. With `anthropic_api`, the parts up to the
  second-to-last are cached.
- `schema`: a JSON Schema. With it, `Result.output` is the parsed object; without it, the text.
- `effort`: `low`, `medium`, `high`, `xhigh` or `max`, when the model supports it.
- `log_path`: append one JSON line per call to this file, failed calls included.
- `prompt_caching=False`: turns prompt caching off (`DISABLE_PROMPT_CACHING=1` for `claude -p`).
  Cheaper when each input is read only once; see [Spending less](#spending-less).
- `validate`: a function that gets the output and raises `ValueError` to reject it (a value out of
  range, an id that does not exist). The call then fails with `LLMFailed` and the reason is logged.
- `claude_cli`: see [Turning the startup back on](#turning-the-startup-back-on).

`Result` has `output`, `model` (the one that answered), `prompt_version`, `backend`,
`input_tokens` (all input, cached or not), `output_tokens`, `cache_read_tokens`,
`cache_write_tokens`, `cost_usd`, `turns` and `seconds`. `cost_usd` is what `claude -p` reports at
list price, which is not what a subscription charges but is good for comparing variants; the API
backend leaves it `None`.

`ask()` is synchronous. From async code, use `await asyncio.to_thread(ask, ...)`.

### Turning the startup back on

The minimal startup suits calls where the whole input fits in the message. When the model needs to
look around, pass `ClaudeCliOptions`:

```python
from llm_ask import ClaudeCliOptions, ask

result = ask(
    "List the TODO comments in this project.",
    system="You review code.",
    model="sonnet",
    step="todos",
    prompt_version="todos-v1",
    claude_cli=ClaudeCliOptions(tools=("Read", "Grep", "Glob"), cwd="path/to/project", max_budget_usd=0.5),
)
```

| Option | Default | Effect |
|---|---|---|
| `tools` | `()` | Tools the model may use without asking, such as `("Read", "Grep")`; MCP tools as `"mcp__server__tool"`. Everything else is denied (`--permission-mode dontAsk`), so the call never waits for an answer. |
| `cwd` | empty temporary directory | Where the CLI runs: what the tools see, and where `setting_sources="project"` reads from. |
| `setting_sources` | `""` | Settings files to load: `user`, `project`, `local`, comma-separated. |
| `mcp_config` | `None` | MCP servers as a JSON file path or JSON string. Only these load, never the user's own. |
| `slash_commands` | `False` | Allow skills and slash commands, from the settings sources loaded. |
| `max_budget_usd` | `None` | Stop the call past this list-price cost. Worth setting with tools, since each tool use is another turn. |

The session is never saved. With `anthropic_api`, `tools` or `mcp_config` raise `ValueError`,
since that backend cannot run them; the other options are ignored.

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

## Spending less

Measured with `claude -p` and Haiku; check them on your own calls, since the `turns` and token
columns of the log are there for that.

- **Keep the minimal startup.** The default Claude Code setup sends about 31,500 tokens per turn;
  without MCP servers it was the same; with only `Read` and no MCP servers or skills, about 5,100;
  with your own system prompt too, about 2,300. `ask()` does this by default.
- **Put the whole input in the message.** Each file the model opens with a tool adds a turn that
  re-reads the whole context. In one app a single-image call reached about 73,000 tokens that way.
- **Turn the cache off for inputs read once** (`prompt_caching=False`). With a subscription,
  writing to the prompt cache costs about twice normal input, and data that is never sent again
  is never read back. In one app this cut about 30% per call with the same answers. The shared
  prefix (system prompt, schema) is still read from the cache. Leave caching on when the same
  large input goes into several calls within the hour.
- **Many calls with the same prefix: run the first alone, then the rest in parallel.** The first
  call writes the prefix to the cache; the others read it.
- **Keep `--json-schema`.** Structured output costs one extra turn, but asking for JSON in the text
  instead only saved about 5% of input, was slower and was less accurate.
- **Choose the model per task, by measuring.** Compare variants with the log's tokens, `cost_usd`
  and `turns`, not by feel.
- **What did not help:** `--effort low` changed nothing measurable on Haiku, and its reasoning
  could not be turned off (`MAX_THINKING_TOKENS=0`). `--exclude-dynamic-system-prompt-sections`
  changes nothing with your own system prompt.

### Answering faster

These are not in `ask()`: they need a long-running process and streaming, which only pay off in
real time apps (a voice bot, for instance).

- **Starting `claude -p` costs about 1.7 to 5 seconds.** Keep it off the critical path.
- **Keep one conversation alive** with `--input-format stream-json --output-format stream-json`
  and write each request as a new user message. A decision took about 1.5 s instead of 3.7 s, as
  it skips the startup and reuses the conversation's cache. The context grows with each message.
- **Keep processes started and waiting** on stdin, one per (model, effort, system prompt), and
  use each for a single answer. The first streamed token arrived in about 0.7 s instead of 5 s.
- **Reasoning:** in one app, `MAX_THINKING_TOKENS=0` with `--effort low` cut a decision from about
  2.3 s to 1 s, while on Haiku it made no difference (see above). Measure it on your model.

### Other tips

- **Images:** scale them to 1568 px on the long side before sending. More adds tokens, not detail.
  (`ask()` does not take images yet.)
- **A separate login for the app:** `claude setup-token` creates a long-lived token. Pass it in
  `CLAUDE_CODE_OAUTH_TOKEN` and the app's calls do not use or refresh the terminal's login. `ask()`
  passes the environment through.

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
