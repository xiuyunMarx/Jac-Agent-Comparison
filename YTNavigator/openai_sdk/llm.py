"""The model seam: one call to `chat.completions.create`, recorded.

This is the whole of what the other two sides get from a framework. byLLM
writes `by router_llm(temperature=0.0)` on a function and the compiler builds
the request, the schema handling and (via a litellm success callback plus a
settle loop) the accounting; LangGraph writes `init_chat_model(...)` and reads
`usage_metadata` through a callback handler. Here the request is written out,
and usage arrives on the response object on this thread, so the per-call
record is complete the moment `complete()` returns -- no callback, no settle
loop.

Models: both roles (INSTANT / POWERFUL) are hard-coded to ollama_chat/glm-5.2:cloud,
at temperature 0; no env override.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

# $BENCH_MODEL is the one knob every arm reads (bare id, default glm-5.2).
DEFAULT_INSTANT_MODEL = os.environ.get("BENCH_MODEL", "glm-5.2")
DEFAULT_POWERFUL_MODEL = os.environ.get("BENCH_MODEL", "glm-5.2")
# Pinned identically on all three sides, or the benchmark measures the model
# rather than the framework.
TEMPERATURE = 0.0

_client: Any | None = None

# Every call this process makes, in order, in the shared benchmark schema's
# llm_calls element shape. main.py slices it per question.
_calls: list[dict[str, Any]] = []


def normalize_model(name: str) -> str:
    """The bare model id, as the raw OpenAI SDK wants it.

    Drops the litellm provider segment ("ollama_chat/glm-5.2:cloud" ->
    "glm-5.2:cloud"; the ":tag" is kept), because `chat.completions.create`
    takes a bare model name and routes by base URL instead. The stderr note
    below says which OPENAI_BASE_URL that provider needs rather than letting
    an unrouted name fail at the provider.
    """
    name = name.strip()
    provider, sep, bare = name.partition("/")
    if not sep:
        return name
    if provider.lower() != "openai" and not os.environ.get("OPENAI_BASE_URL"):
        sys.stderr.write(
            f"[llm] note: model '{name}' names provider '{provider}', but this side talks to "
            "whatever OPENAI_BASE_URL points at (default: OpenAI). Set OPENAI_BASE_URL to that "
            f"provider's OpenAI-compatible endpoint; calling '{bare}' as-is.\n"
        )
    return bare


def instant_model() -> str:
    """The router / direct-reply model (the original's settings.INSTANT_LLM)."""
    return normalize_model(DEFAULT_INSTANT_MODEL)


def powerful_model() -> str:
    """The tool agent's model (the original's settings.POWERFUL_LLM)."""
    return normalize_model(DEFAULT_POWERFUL_MODEL)


def build_client() -> Any:
    # Imported here rather than at module scope so `import llm` works with no
    # `openai` installed and no key set. OPENAI_API_KEY and OPENAI_BASE_URL
    # are read natively by the SDK; deliberately not named here, so an unset
    # variable fails at the provider with the provider's own message.
    from openai import OpenAI

    return OpenAI()


def get_client() -> Any:
    global _client
    if _client is None:
        _client = build_client()
    return _client


def set_client(client: Any | None) -> None:
    """Swap the client (tests, stand-ins). Pass None to fall back."""
    global _client
    _client = client


def llm_call_count() -> int:
    """Marker for llm_calls_since -- same pair of helpers byLLM's tracker has."""
    return len(_calls)


def llm_calls_since(index: int) -> list[dict[str, Any]]:
    """Per-call records made since the marker. No settle loop: usage came back
    on the response, so the log was complete before the agent returned."""
    return list(_calls[index:])


def complete(
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str,
    tools: Iterable[Mapping[str, Any]] | None = None,
    client: Any | None = None,
) -> Any:
    """One round trip. Returns the assistant message; records the call."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": TEMPERATURE,
    }
    tool_specs = list(tools) if tools else None
    if tool_specs:
        payload["tools"] = tool_specs

    api = (client or get_client()).chat.completions
    started = time.perf_counter()
    response = api.create(**payload)
    latency = round(time.perf_counter() - started, 4)

    usage = getattr(response, "usage", None)
    _calls.append(
        {
            "model": getattr(response, "model", None) or model,
            "latency_s": latency,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        }
    )
    return response.choices[0].message


def assistant_turn(message: Any) -> dict[str, Any]:
    """The assistant message as a plain dict, ready to send back as history."""
    turn: dict[str, Any] = {"role": "assistant", "content": message.content}
    calls = getattr(message, "tool_calls", None) or []
    if calls:
        turn["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in calls
        ]
    elif turn["content"] is None:
        turn["content"] = ""
    return turn


def tool_turn(call_id: str, name: str, content: str) -> dict[str, Any]:
    """One tool result, in the shape the API wants it answered."""
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


__all__ = [
    "DEFAULT_INSTANT_MODEL",
    "DEFAULT_POWERFUL_MODEL",
    "TEMPERATURE",
    "assistant_turn",
    "build_client",
    "complete",
    "get_client",
    "instant_model",
    "llm_call_count",
    "llm_calls_since",
    "normalize_model",
    "powerful_model",
    "set_client",
    "tool_turn",
]
