from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest

from llm_ask import LLMFailed, LLMUnavailable, ask
from llm_ask import anthropic_api

SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def message(text='{"n": 3}', stop_reason="end_turn"):
    usage = SimpleNamespace(input_tokens=2, cache_read_input_tokens=50, cache_creation_input_tokens=100, output_tokens=7)
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason,
                           model="claude-haiku-5-5", usage=usage)


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("LLM_ASK_BACKEND", "anthropic_api")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    class FakeCreate:
        reply: object = message()
        params: list[dict] = []

        def __call__(self, p, timeout):
            self.params.append(p)
            if isinstance(self.reply, BaseException):
                raise self.reply
            return self.reply

    fake = FakeCreate()
    fake.params = []
    monkeypatch.setattr(anthropic_api, "_create", fake)
    return fake


def ask_n(**kwargs):
    return ask(["stable", "variable"], system="Count.", model="claude-haiku-5-5", step="count",
               prompt_version="v1", schema=SCHEMA, **kwargs)


def test_request_params(api):
    ask_n(effort="low")
    p = api.params[0]
    assert p["system"] == "Count."
    assert p["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}
    first, last = p["messages"][0]["content"]
    assert first["cache_control"] == {"type": "ephemeral"} and "cache_control" not in last


def test_no_cache_marker_when_caching_is_off(api):
    ask_n(prompt_caching=False)
    assert all("cache_control" not in block for block in api.params[0]["messages"][0]["content"])


def test_result(api):
    result = ask_n()
    assert (result.output, result.model, result.backend) == ({"n": 3}, "claude-haiku-5-5", "anthropic_api")
    assert (result.input_tokens, result.output_tokens) == (152, 7)


def test_missing_key_means_unavailable(api, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(LLMUnavailable):
        ask_n()


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (anthropic.APIConnectionError(request=REQUEST), LLMUnavailable),
        (anthropic.AuthenticationError("bad key", response=httpx.Response(401, request=REQUEST), body=None), LLMUnavailable),
        (anthropic.InternalServerError("oops", response=httpx.Response(500, request=REQUEST), body=None), LLMFailed),
        (message(stop_reason="refusal"), LLMFailed),
        (message(stop_reason="max_tokens"), LLMFailed),
        (message(text="not json"), LLMFailed),
    ],
)
def test_failures_raise_typed_errors(api, reply, error):
    api.reply = reply
    with pytest.raises(error):
        ask_n()
