"""Benchmark instrumentation for the agent.

Two pieces:

- A contextvar-based trace that agent code writes *fallback events* to
  (silent error-recovery paths that would otherwise be indistinguishable
  from normal answers in benchmark results). Outside an active trace every
  call is a no-op, so production behavior is unchanged.
- A LangChain callback handler that records per-LLM-call latency and token
  usage, and per-tool-call latency, for the benchmark runner.
"""

import time
from contextvars import ContextVar
from typing import (
    Any,
    Dict,
    List,
    Optional,
)
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult


class BenchmarkTrace:
    """Collects events recorded during a single agent invocation."""

    def __init__(self):
        """Initialize an empty trace."""
        self.events: List[Dict[str, Any]] = []

    def record(self, event: str, **data: Any) -> None:
        """Append an event to the trace.

        Args:
            event: Short event name, e.g. "router_parse_fallback".
            **data: Arbitrary JSON-serializable context for the event.
        """
        self.events.append({"event": event, **data})


_active_trace: ContextVar[Optional[BenchmarkTrace]] = ContextVar("benchmark_trace", default=None)


def start_trace() -> BenchmarkTrace:
    """Activate a fresh trace for the current context and return it."""
    trace = BenchmarkTrace()
    _active_trace.set(trace)
    return trace


def stop_trace() -> None:
    """Deactivate the current trace."""
    _active_trace.set(None)


def record_event(event: str, **data: Any) -> None:
    """Record an event on the active trace, if any.

    No-op when no trace is active (i.e. outside benchmark runs).

    Args:
        event: Short event name.
        **data: Arbitrary JSON-serializable context for the event.
    """
    trace = _active_trace.get()
    if trace is not None:
        trace.record(event, **data)


class BenchmarkCallbackHandler(AsyncCallbackHandler):
    """Records per-LLM-call and per-tool-call timing and token usage.

    Pass an instance via ``AgentGraph.invoke(..., callbacks=[handler])`` and read
    ``handler.llm_calls`` / ``handler.tool_calls`` after the run.
    """

    def __init__(self):
        """Initialize empty call records."""
        self.llm_calls: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self._pending: Dict[UUID, Dict[str, Any]] = {}

    @staticmethod
    def _extract_usage(response: LLMResult) -> Dict[str, Optional[int]]:
        """Pull token usage out of an LLMResult, handling both Groq layouts."""
        usage = (response.llm_output or {}).get("token_usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        if prompt_tokens is None:
            for generations in response.generations:
                for generation in generations:
                    message = getattr(generation, "message", None)
                    usage_metadata = getattr(message, "usage_metadata", None)
                    if usage_metadata:
                        prompt_tokens = usage_metadata.get("input_tokens")
                        completion_tokens = usage_metadata.get("output_tokens")
                        break

        return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

    async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        """Record the start time and model name of a chat model call."""
        model = (serialized or {}).get("kwargs", {}).get("model_name")
        self._pending[run_id] = {"model": model, "started": time.perf_counter()}

    async def on_llm_end(self, response: LLMResult, *, run_id, **kwargs):
        """Record latency and token usage when an LLM call finishes."""
        pending = self._pending.pop(run_id, {})
        self.llm_calls.append(
            {
                "model": pending.get("model"),
                "latency_s": round(time.perf_counter() - pending["started"], 4) if pending else None,
                **self._extract_usage(response),
            }
        )

    async def on_llm_error(self, error, *, run_id, **kwargs):
        """Record a failed LLM call."""
        pending = self._pending.pop(run_id, {})
        self.llm_calls.append(
            {
                "model": pending.get("model"),
                "latency_s": round(time.perf_counter() - pending["started"], 4) if pending else None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "error": str(error),
            }
        )

    async def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        """Record the start time and name of a tool call."""
        name = (serialized or {}).get("name")
        self._pending[run_id] = {"name": name, "started": time.perf_counter()}

    async def on_tool_end(self, output, *, run_id, **kwargs):
        """Record latency when a tool call finishes."""
        pending = self._pending.pop(run_id, {})
        self.tool_calls.append(
            {
                "name": pending.get("name"),
                "latency_s": round(time.perf_counter() - pending["started"], 4) if pending else None,
            }
        )

    async def on_tool_error(self, error, *, run_id, **kwargs):
        """Record a failed tool call."""
        pending = self._pending.pop(run_id, {})
        self.tool_calls.append(
            {
                "name": pending.get("name"),
                "latency_s": round(time.perf_counter() - pending["started"], 4) if pending else None,
                "error": str(error),
            }
        )

    def total_tokens(self) -> Dict[str, int]:
        """Sum token usage across all recorded LLM calls."""
        prompt = sum(c["prompt_tokens"] or 0 for c in self.llm_calls)
        completion = sum(c["completion_tokens"] or 0 for c in self.llm_calls)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
