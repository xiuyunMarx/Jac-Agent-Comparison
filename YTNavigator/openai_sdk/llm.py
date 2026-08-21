"""The model seam: one call to `chat.completions.create`, recorded.

This is the whole of what the other two sides get from a framework. byLLM
writes `by router_llm(temperature=0.0)` on a function and the compiler builds
the request, the schema handling and (via a litellm success callback plus a
settle loop) the accounting; LangGraph writes `init_chat_model(...)` and reads
`usage_metadata` through a callback handler. Here the request is written out,
and usage arrives on the response object on this thread, so the per-call
record is complete the moment `complete()` returns -- no callback, no settle
loop.

Models are the original app's pair: INSTANT_LLM (default gpt-4o-mini) for
routing and direct replies, POWERFUL_LLM (default gpt-4o) for the tool agent,
both at temperature 0. The same env values configure all three sides:
langchain-style "provider:model" and litellm-style "provider/model" are both
accepted; the "openai" prefix is stripped for the raw SDK, and any other
provider's bare model name is used against OPENAI_BASE_URL (every provider the
other two sides can reach speaks the OpenAI wire format at some base URL).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_INSTANT_MODEL = "gpt-4o-mini"
DEFAULT_POWERFUL_MODEL = "gpt-4o"
# Pinned identically on all three sides, or the benchmark measures the model
# rather than the framework.
TEMPERATURE = 0.0

_client: Any | None = None

# Every call this process makes, in order, in the shared benchmark schema's
# llm_calls element shape. main.py slices it per question.
_calls: list[dict[str, Any]] = []


def normalize_model(name: str) -> str:
    """A provider-agnostic model value, as the raw OpenAI SDK wants it.

    byLLM normalizes "provider:model" to litellm's "provider/model" with one
    replace; this side goes one step further and drops the provider segment,
    because `chat.completions.create` takes a bare model name and routes by
    base URL instead. "gpt-4o-mini", "openai:gpt-4o-mini" and
    "openai/gpt-4o-mini" all mean the same model; "groq:llama-3.1-8b-instant"
    means llama-3.1-8b-instant at whatever OPENAI_BASE_URL points to (Groq's
    OpenAI-compatible endpoint, for that value), and the stderr note below
    says so rather than letting an unrouted name fail at the provider.
    """
    name = name.strip().replace(":", "/", 1)
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
    return normalize_model(os.environ.get("INSTANT_LLM", DEFAULT_INSTANT_MODEL))


def powerful_model() -> str:
    """The tool agent's model (the original's settings.POWERFUL_LLM)."""
    return normalize_model(os.environ.get("POWERFUL_LLM", DEFAULT_POWERFUL_MODEL))


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
