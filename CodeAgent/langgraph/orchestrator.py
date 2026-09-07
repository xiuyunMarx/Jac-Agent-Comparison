"""SWE-bench result adapter and optional LangChain callback telemetry."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import convert_to_openai_messages

import main
from nodes import MODEL_NAME, tool_log

usage_log: list[dict] = []


class RunTelemetry(BaseCallbackHandler):
    def __init__(self):
        self.requests = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        if os.environ.get("CODEAGENT_TRACE"):
            self.requests[run_id] = {
                **kwargs["invocation_params"],
                "messages": convert_to_openai_messages(messages[0]),
            }

    def on_llm_end(self, response, *, run_id, **kwargs):
        message = response.generations[0][0].message
        usage = message.usage_metadata or {}
        totals = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "prompt_tokens_details": {"cached_tokens": usage.get("input_token_details", {}).get("cache_read", 0)},
        }
        usage_log.append(totals)
        payload = self.requests.pop(run_id, None)
        if payload is not None:
            row = {"call": len(usage_log), "model": payload.get("model"),
                   "n_messages": len(payload["messages"]), "messages": payload["messages"],
                   "tools": [tool["function"]["name"] for tool in payload.get("tools", [])],
                   "tool_schemas": payload.get("tools"), "temperature": payload.get("temperature"),
                   "response_format": payload.get("response_format"), "extra": [], "last": None,
                   "usage": totals, "reply": convert_to_openai_messages(message)}
            with open(os.environ["CODEAGENT_TRACE"], "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.requests.pop(run_id, None)


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


def active_model_name() -> str:
    return MODEL_NAME


def solve(objective: str, repo_root: str = "", max_steps: int = 10) -> RunResult:
    usage_log.clear()
    answer = main.solve(objective, repo_root or ".", config={"callbacks": [RunTelemetry()]})
    usage = list(usage_log)
    calls = [ToolCall(name=str(c["name"]), args={"args": str(c["args"]), "ok": str(c["ok"])})
             for c in tool_log]
    return RunResult(
        objective=objective,
        answer=answer,
        steps=len(re.findall(r"(?m)^### (Plan|Explore|Edit|Verify)\b", answer)),
        tool_calls=calls,
        llm_calls=len(usage),
        prompt_tokens=sum(u["prompt_tokens"] for u in usage),
        completion_tokens=sum(u["completion_tokens"] for u in usage),
        cached_tokens=sum(u["prompt_tokens_details"]["cached_tokens"] for u in usage),
    )


__all__ = ["DEFAULT_MODEL", "RunResult", "ToolCall", "active_model_name", "solve"]
