"""The model seam: one call to `chat.completions.create`, recorded.

This is the whole of what the other two sides get from a framework. byLLM writes
`by agent_model(...)` on a function and the compiler produces the request, the
tool schemas, the loop and the accounting; LangGraph writes
`ChatOpenAI(...).bind_tools(...)` and gets the request and the accounting. Here
the request is written out.

The client is constructed lazily and reachable only through this seam, so
importing the agent needs no API key -- the topology and the capability boundary
can be inspected without credentials, and a caller can swap in a stand-in that
never touches the network.
"""

from __future__ import annotations

import sys
import os
from typing import Any, Iterable, Mapping, Sequence

from telemetry import TokenUsage

# Pinned identically on all three sides of the comparison, or the benchmark
# measures the model rather than the framework. Overridable per shell with
# CODEAGENT_MODEL, exactly as byLLM/jac.toml and langgraph/pyproject.toml do it.
DEFAULT_MODEL = "gpt-5"
# byLLM's [byllm.call_params], and gpt-5-shaped throughout.
#
# temperature 1: gpt-5 rejects every other value ("Unsupported value:
# 'temperature' does not support 0.0 with this model"). The 0.0 this used to
# send bought no reproducibility on a reasoning model and only made the config
# disagree with the run.
#
# max_tokens 16384: a reasoning model bills its hidden reasoning against the
# completion budget. One trivial turn measured 3264 reasoning tokens against
# 1459 of visible content; at the old 4096 the model stops mid-tool-call, which
# corrupts an edit rather than merely shortening prose.
TEMPERATURE = float(os.environ.get("CODEAGENT_TEMPERATURE", "1.0"))
MAX_TOKENS = int(os.environ.get("CODEAGENT_MAX_TOKENS", "16384"))
# Streamed, to match byLLM's run_phase. `_relax` still covers a provider that
# refuses either parameter outright -- but not one that refuses
# `stream_options.include_usage`, which streaming always sends and which some
# OpenAI-compatible servers reject. $CODEAGENT_STREAM=0 for those.
STREAM = os.environ.get("CODEAGENT_STREAM", "1") not in ("0", "false", "no")
# How much of a streamed tool call is echoed to stderr. The full arguments are
# already in the call log; repeating them here would drown the transcript.
STREAM_PREVIEW_CHARS = 160

_client: Any | None = None
_model: str | None = None

# Every call this process makes, in order. `solve` resets it per run; `llm_log`
# renders it.
token_usage = TokenUsage()


def build_client() -> Any:
    # Imported here rather than at module scope so that `import llm` works with
    # no `openai` installed and no key set -- the same reason LangGraph's side
    # defers `from langchain_openai import ChatOpenAI` into build_model().
    from openai import OpenAI

    # OPENAI_API_KEY and OPENAI_BASE_URL are read natively; deliberately not
    # named here, so an unset variable fails at the provider with the provider's
    # own message rather than being passed through as a literal.
    return OpenAI()


def get_client() -> Any:
    global _client
    if _client is None:
        _client = build_client()
    return _client


def set_client(client: Any | None) -> None:
    """Swap the client. Pass None to fall back to the configured provider."""
    global _client
    _client = client


def active_model_name() -> str:
    """The model this run will call.

    Reads the environment rather than the client, so it answers before any
    client exists -- the SWE-bench shim asks for it while reporting a failed run,
    where constructing a provider would raise and lose the report.
    """
    # byLLM and CrewAI need litellm's "openai/" prefix on a name litellm does
    # not know; the raw SDK wants the bare id, so a provider prefix is stripped
    # rather than passed through. One $CODEAGENT_MODEL, three shapes.
    name = _model or os.environ.get("CODEAGENT_MODEL", DEFAULT_MODEL)
    return name.replace(":", "/").split("/")[-1]


def set_model(model: str | None) -> None:
    global _model
    _model = model


def assistant_turn(message: Any) -> dict[str, Any]:
    """The assistant message as a plain dict, ready to send back as history.

    Plain dicts rather than the SDK's own objects throughout: the conversation
    is then a value the phase loop can slice, print and rebuild without the
    provider's types leaking into the orchestration.
    """
    turn: dict[str, Any] = {"role": "assistant", "content": message.content}
    calls = getattr(message, "tool_calls", None) or []
    if calls:
        turn["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in calls
        ]
    elif turn["content"] is None:
        turn["content"] = ""
    return turn


def tool_turn(call_id: str, name: str, content: str) -> dict[str, Any]:
    """One tool result, in the shape the API wants it answered."""
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}

def _relax(payload: dict[str, Any], error: str) -> dict[str, Any] | None:
    text = error.lower()
    if "max_tokens" in text and "max_completion_tokens" in text and "max_tokens" in payload:
        out = dict(payload)
        out["max_completion_tokens"] = out.pop("max_tokens")
        return out
    if "temperature" in text and "temperature" in payload:
        out = dict(payload)
        out.pop("temperature")
        return out
    return None


def _emit(text: str) -> None:
    """Live output goes to stderr, as byLLM's stream handler does.

    stdout is a transcript rather than a channel on every side of this
    comparison -- the SWE-bench shim parses files, not console output -- and the
    driver captures both streams anyway.
    """
    sys.stderr.write(text)
    sys.stderr.flush()


def _dispatch(api: Any, payload: Mapping[str, Any]) -> Any:
    """One round trip, streamed or not, returning a whole ChatCompletion.

    `api.stream()` accumulates the deltas itself and `get_final_completion()`
    returns the same shape `api.create()` does -- choices[0].message with its
    tool_calls, and .usage -- so nothing downstream of `complete` needs to know
    whether the call streamed. Hand-accumulating tool_call deltas by index is
    the alternative, and it is exactly the bookkeeping that silently truncates
    an edit when one corner is got wrong.
    """
    if not STREAM:
        return api.create(**payload)
    # include_usage is not optional bookkeeping: without it a streamed
    # completion comes back with usage=None, every call records 0 prompt and 0
    # completion tokens, and the run reports a cost of nothing at all. Measured
    # -- the first streamed build of this function did exactly that.
    payload = {**payload, "stream_options": {"include_usage": True}}
    with api.stream(**payload) as stream:
        for event in stream:
            if getattr(event, "type", "") == "content.delta":
                _emit(str(getattr(event, "delta", "") or ""))
        final = stream.get_final_completion()
    message = final.choices[0].message if final.choices else None
    for call in list(getattr(message, "tool_calls", None) or []):
        fn = getattr(call, "function", None)
        args = str(getattr(fn, "arguments", "") or "")
        if len(args) > STREAM_PREVIEW_CHARS:
            args = args[:STREAM_PREVIEW_CHARS] + "..."
        _emit(f"\n  -> {getattr(fn, 'name', '?')}({args})\n")
    return final


def complete(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Iterable[Mapping[str, Any]] | None = None,
    response_format: Mapping[str, Any] | None = None,
    model: str | None = None,
    usage: TokenUsage | None = None,
    client: Any | None = None,
) -> Any:
    """One round trip. Returns the assistant message; records the call.

    Usage comes back on the response, on this thread, so the record is complete
    the moment this returns -- see the module docstring of telemetry.py for what
    that deletes relative to byLLM.
    """
    usage = token_usage if usage is None else usage
    name = model or active_model_name()
    tool_specs = list(tools) if tools else None
    payload: dict[str, Any] = {
        "model": name,
        "messages": list(messages),
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    if tool_specs:
        payload["tools"] = tool_specs
    if response_format is not None:
        payload["response_format"] = dict(response_format)

    api = (client or get_client()).chat.completions
    try:
        response = _dispatch(api, payload)
    except Exception as e:  # noqa: BLE001 - inspected, then re-raised if unknown
        relaxed = _relax(payload, str(e))
        if relaxed is None:
            raise
        response = _dispatch(api, relaxed)

    usage.track(model=name, messages=payload["messages"], tools=tool_specs, response=response)
    return response.choices[0].message


__all__ = [
    "DEFAULT_MODEL",
    "MAX_TOKENS",
    "STREAM",
    "TEMPERATURE",
    "active_model_name",
    "assistant_turn",
    "build_client",
    "complete",
    "get_client",
    "set_client",
    "set_model",
    "token_usage",
    "tool_turn",
]
