import json

import pytest

from llm_ask import LLMFailed, LLMUnavailable, ask
from llm_ask.pytest_plugin import RealLLMCallBlocked


def ask_simple(**kwargs):
    return ask("Hi", system="Greet.", model="haiku", step="greet", prompt_version="v2", **kwargs)


def test_real_calls_are_blocked_in_tests():
    with pytest.raises(RealLLMCallBlocked):
        ask_simple()


def test_real_api_calls_are_blocked_in_tests(monkeypatch):
    monkeypatch.setenv("LLM_ASK_BACKEND", "anthropic_api")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with pytest.raises(RealLLMCallBlocked):
        ask_simple()


def test_unknown_backend(monkeypatch):
    monkeypatch.setenv("LLM_ASK_BACKEND", "other")
    with pytest.raises(ValueError):
        ask_simple()


def test_fake_llm_answers_and_records(fake_llm):
    fake_llm.respond("hello", LLMUnavailable("down"))
    assert ask_simple().output == "hello"
    with pytest.raises(LLMUnavailable):
        ask_simple()
    assert [r.parts for r in fake_llm.requests] == [["Hi"], ["Hi"]]


def test_log_has_one_line_per_call_including_failures(fake_llm, tmp_path):
    log = tmp_path / "logs" / "llm.jsonl"
    fake_llm.respond("hello", LLMFailed("refused"))
    ask_simple(log_path=log)
    with pytest.raises(LLMFailed):
        ask_simple(log_path=log)
    ok, failed = [json.loads(line) for line in log.read_text().splitlines()]
    assert {k: ok[k] for k in ("step", "backend", "model", "prompt_version", "input_tokens", "output_tokens")} == {
        "step": "greet", "backend": "claude_cli", "model": "haiku", "prompt_version": "v2",
        "input_tokens": 0, "output_tokens": 0,
    }
    assert "seconds" in ok and "ts" in ok and "error" not in ok
    assert failed["error"] == "LLMFailed: refused"
