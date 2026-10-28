"""pytest plugin, loaded automatically wherever llm-ask is installed.

- Every test runs with real LLM calls blocked: reaching `claude -p` or the Anthropic API raises
  RealLLMCallBlocked. It is not an LLMError on purpose, so the app's fallback does not hide it.
- The `fake_llm` fixture answers `ask()` with canned outputs and records the requests.
"""

import pytest

from . import anthropic_api, claude_cli, core


class RealLLMCallBlocked(RuntimeError):
    pass


def _blocked(*args, **kwargs):
    raise RealLLMCallBlocked("a test tried to call a real LLM; use the fake_llm fixture")


@pytest.fixture(autouse=True)
def _llm_ask_block_real_calls(monkeypatch):
    monkeypatch.setattr(claude_cli, "_run", _blocked)
    monkeypatch.setattr(anthropic_api, "_create", _blocked)


class FakeLLM:
    def __init__(self):
        self.outputs: list = []
        self.requests: list[core.Request] = []

    def respond(self, *outputs) -> None:
        """Queues the outputs of the next calls, in order. An exception instance is raised instead."""
        self.outputs.extend(outputs)

    def _call(self, backend: str, request: core.Request) -> core.Reply:
        self.requests.append(request)
        if not self.outputs:
            raise AssertionError("fake_llm: ask() was called with no queued output; use fake_llm.respond()")
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return core.Reply(output=output, model=request.model, input_tokens=0, output_tokens=0)


@pytest.fixture
def fake_llm(monkeypatch) -> FakeLLM:
    fake = FakeLLM()
    monkeypatch.setattr(core, "_call_backend", fake._call)
    return fake
