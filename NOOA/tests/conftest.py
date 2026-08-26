"""Fixtures: a temporary repository, and a model that never leaves the process.

The agent home goes on `sys.path` the way ../swebench_bridge/swe_entry.py does
it, so the tests import `orchestrator` under the same module names the benchmark
will.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HOME not in sys.path:
    sys.path.insert(0, HOME)

from nooa.unifiedllm import FakeLLMClient  # noqa: E402
from nooa.unifiedllm.unifiedllm import LLMResponse, ToolCall  # noqa: E402

# What one scripted turn reports as its cost. Fixed rather than derived, so a
# test can assert the totals arithmetic exactly.
FAKE_USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "total_tokens": 120,
    "prompt_tokens_details": {"cached_tokens": 40},
}


def cell(code: str, call_id: str = "c") -> LLMResponse:
    """One scripted turn: a CodeAct `execute_python` call carrying `code`.

    Every turn in these tests is a code cell, because `return_result` is
    callable from inside one -- so a session ends by running
    `return_result(value)` in its last cell rather than through a second tool.
    """
    tool_call = ToolCall(
        id=call_id, name="execute_python", arguments=json.dumps({"code": code})
    )
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[tool_call],
        finish_reason="tool_calls",
        assistant_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
            ],
        },
        usage=dict(FAKE_USAGE),
    )


def scripted(*codes: str) -> FakeLLMClient:
    """A model that answers with these cells, in this order, and then nothing."""
    return FakeLLMClient(
        scripted_responses=[cell(c, f"c{i}") for i, c in enumerate(codes)]
    )


def report_cell(**fields: object) -> str:
    """A cell that finishes the session by returning a Report."""
    base = {
        "resolved": True,
        "summary": "fixed it",
        "files_changed": ["pkg/greet.py"],
        "verify_command": "git status",
        "evidence": "exit_code: 0",
    }
    base.update(fields)
    return f"return_result({base!r})"


@pytest.fixture
def repo(tmp_path):
    """A tiny git repository with one module in it."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "greet.py").write_text('def greet():\n    return "hello"\n')
    (root / "README.md").write_text("# demo\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


@pytest.fixture(autouse=True)
def clean_state():
    """Reset the two pieces of module-level state a run owns."""
    from llm import set_client, set_model, token_usage
    from tools.common import reset_tool_log

    reset_tool_log()
    token_usage.reset()
    yield
    set_client(None)
    set_model(None)
    reset_tool_log()
    token_usage.reset()
