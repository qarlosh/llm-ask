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
    assert {k: v for k, v in ok.items() if k not in ("ts", "seconds")} == {
        "step": "greet", "backend": "claude_cli", "model": "haiku", "prompt_version": "v2",
        "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0,
        "cost_usd": None, "turns": 1,
    }
    assert "seconds" in ok and "ts" in ok and "error" not in ok
    assert failed["error"] == "LLMFailed: refused"


def _positive(output):
    if output["n"] <= 0:
        raise ValueError(f"n must be positive, got {output['n']}")


def test_validate_rejects_and_logs(fake_llm, tmp_path):
    log = tmp_path / "llm.jsonl"
    fake_llm.respond({"n": 3}, {"n": -1})
    assert ask_simple(validate=_positive, log_path=log).output == {"n": 3}
    with pytest.raises(LLMFailed, match="n must be positive"):
        ask_simple(validate=_positive, log_path=log)
    assert json.loads(log.read_text().splitlines()[1])["error"] == "LLMFailed: output rejected: n must be positive, got -1"
