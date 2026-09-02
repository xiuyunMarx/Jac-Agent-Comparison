"""The swebench_bridge adapter: swe_entry.py imports solve / active_model_name / DEFAULT_MODEL from here.

The twin of ../Jac/orchestrator.jac, and the same file as ../openai_sdk/orchestrator.py:
both Python sides put usage into `nodes.usage_log` in the OpenAI usage shape,
so the totals are computed by one function on every side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import main
from nodes import MODEL_NAME, llm, tool_log, usage_log

DEFAULT_MODEL: str = MODEL_NAME


@dataclass
class ToolCall:
    name: str
    args: dict[str, str] = field(default_factory=dict)


@dataclass
class RunResult:
    objective: str
    answer: str = ""
    steps: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


def _usage_int(u: Any, key: str) -> int:
    try:
        if isinstance(u, dict):
            return int(u.get(key, 0) or 0)
        return int(getattr(u, key, 0) or 0)
    except Exception:  # noqa: BLE001
        return 0


def _cached_tokens(u: Any) -> int:
    n = _usage_int(u, "cache_read_input_tokens")
    if n:
        return n
    details = u.get("prompt_tokens_details") if isinstance(u, dict) else getattr(u, "prompt_tokens_details", None)
    if details is None:
        return 0
    return _usage_int(details, "cached_tokens")


def active_model_name() -> str:
    return str(llm.model_name)


def solve(objective: str, repo_root: str = "", max_steps: int = 10) -> RunResult:
    usage_log.clear()
    answer = main.solve(objective, repo_root or ".")
    usage = list(usage_log)
    calls = [ToolCall(name=str(c["name"]), args={"args": str(c["args"]), "ok": str(c["ok"])})
             for c in tool_log]
    return RunResult(
        objective=objective,
        answer=answer,
        steps=len(re.findall(r"(?m)^### (Plan|Explore|Edit|Verify)\b", answer)),
        tool_calls=calls,
        llm_calls=len(usage),
        prompt_tokens=sum(_usage_int(u, "prompt_tokens") for u in usage),
        completion_tokens=sum(_usage_int(u, "completion_tokens") for u in usage),
        cached_tokens=sum(_cached_tokens(u) for u in usage),
    )


__all__ = ["DEFAULT_MODEL", "RunResult", "ToolCall", "active_model_name", "solve"]
