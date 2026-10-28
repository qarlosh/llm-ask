import json
import os
import subprocess

import pytest

from llm_ask import ClaudeCliOptions, LLMFailed, LLMUnavailable, ask
from llm_ask import claude_cli

SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}


def cli_result(**overrides) -> str:
    result = {
        "type": "result",
        "is_error": False,
        "result": '{"n": 3}',
        "structured_output": {"n": 3},
        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 50, "output_tokens": 7},
        "modelUsage": {"claude-haiku-5-5": {"outputTokens": 7}},
        "total_cost_usd": 0.0004,
        "num_turns": 2,
    }
    return json.dumps(result | overrides)


@pytest.fixture
def fake_run(monkeypatch):
    """Replaces the subprocess call; `fake_run.reply` sets what it returns or raises."""

    class FakeRun:
        reply: object = subprocess.CompletedProcess([], 0, cli_result(), "")
        calls: list[dict] = []

        def __call__(self, cmd, stdin, cwd, env, timeout):
            self.calls.append({"cmd": cmd, "stdin": stdin, "cwd": cwd, "cwd_exists": os.path.isdir(cwd),
                               "env": env, "timeout": timeout})
            if isinstance(self.reply, BaseException):
                raise self.reply
            return self.reply

    fake = FakeRun()
    fake.calls = []
    monkeypatch.setattr(claude_cli, "_run", fake)
    return fake


def ask_n(**kwargs):
    return ask(["stable", "variable"], system="Count.", model="haiku", step="count", prompt_version="v1",
               schema=SCHEMA, **kwargs)


def test_minimal_startup_flags(fake_run):
    ask_n(effort="low")
    call = fake_run.calls[0]
    cmd = call["cmd"]
    assert cmd[:2] == ["claude", "-p"]
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in cmd
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert cmd[cmd.index("--system-prompt") + 1] == "Count."
    assert cmd[cmd.index("--model") + 1] == "haiku"
    assert cmd[cmd.index("--effort") + 1] == "low"
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == SCHEMA
    assert call["stdin"] == "stable\n\nvariable"


def test_runs_in_an_empty_temporary_directory(fake_run):
    ask_n()
    ask_n()
    first, second = fake_run.calls
    assert first["cwd_exists"] and first["cwd"] != second["cwd"]
    assert first["cwd"] != os.getcwd()
    assert not os.path.exists(first["cwd"])  # removed after the call


def test_prompt_caching_can_be_turned_off(fake_run):
    ask_n()
    assert "DISABLE_PROMPT_CACHING" not in fake_run.calls[0]["env"]
    ask_n(prompt_caching=False)
    assert fake_run.calls[1]["env"]["DISABLE_PROMPT_CACHING"] == "1"


def test_minimal_startup_has_no_permission_flags(fake_run):
    ask_n()
    cmd = fake_run.calls[0]["cmd"]
    for flag in ("--allowedTools", "--permission-mode", "--mcp-config", "--max-budget-usd"):
        assert flag not in cmd


def test_options_turn_parts_of_the_startup_back_on(fake_run, tmp_path):
    options = ClaudeCliOptions(
        tools=("Read", "Grep", "mcp__notes__search"),
        cwd=tmp_path,
        setting_sources="project",
        mcp_config='{"mcpServers": {}}',
        slash_commands=True,
        max_budget_usd=0.5,
    )
    ask_n(claude_cli=options)
    call = fake_run.calls[0]
    cmd = call["cmd"]
    assert cmd[cmd.index("--tools") + 1] == "Read,Grep"
    assert cmd[cmd.index("--allowedTools") + 1] == "Read,Grep,mcp__notes__search"
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert cmd[cmd.index("--setting-sources") + 1] == "project"
    assert cmd[cmd.index("--mcp-config") + 1] == '{"mcpServers": {}}'
    assert cmd[cmd.index("--max-budget-usd") + 1] == "0.5"
    assert "--strict-mcp-config" in cmd and "--no-session-persistence" in cmd
    assert "--disable-slash-commands" not in cmd
    assert call["cwd"] == str(tmp_path)


def test_result_carries_output_model_tokens_and_prompt_version(fake_run):
    result = ask_n()
    assert result.output == {"n": 3}
    assert result.model == "claude-haiku-5-5"
    assert result.prompt_version == "v1"
    assert result.backend == "claude_cli"
    assert (result.input_tokens, result.output_tokens) == (152, 7)
    assert (result.cache_read_tokens, result.cache_write_tokens) == (50, 100)
    assert (result.cost_usd, result.turns) == (0.0004, 2)


def test_text_answer_without_schema(fake_run):
    fake_run.reply = subprocess.CompletedProcess([], 0, cli_result(result="hello", structured_output=None), "")
    result = ask("Hi", system="Greet.", model="haiku", step="greet", prompt_version="v1")
    assert result.output == "hello"
    assert "--json-schema" not in fake_run.calls[0]["cmd"]


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (FileNotFoundError("claude"), LLMUnavailable),
        (subprocess.TimeoutExpired("claude", 5), LLMFailed),
        (subprocess.CompletedProcess([], 1, cli_result(is_error=True, result="Not logged in · Please run /login"), ""), LLMUnavailable),
        (subprocess.CompletedProcess([], 1, cli_result(is_error=True, result="API Error: 529 overloaded"), ""), LLMFailed),
        (subprocess.CompletedProcess([], 1, "", "boom"), LLMFailed),
        (subprocess.CompletedProcess([], 1, cli_result(is_error=True, result=None, subtype="error_max_budget_usd",
                                                       errors=["Reached maximum budget ($0.5)"]), ""), LLMFailed),
        (subprocess.CompletedProcess([], 0, cli_result(structured_output=None), ""), LLMFailed),
    ],
)
def test_failures_raise_typed_errors(fake_run, reply, error):
    fake_run.reply = reply
    with pytest.raises(error):
        ask_n()
