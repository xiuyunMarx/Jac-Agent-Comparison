"""The model seam: one call to `chat.completions.create`, metered.

This is the whole of what the other two sides get from a framework. byLLM
writes `by llm(...)` on a function and the compiler produces the request, the
schema and the tool loop; CrewAI hides the same request inside an Agent/Task
pair. Here the request is written out.

Two deliberate choices:

  * The model knob is `OPENAI_MODEL_NAME`, exactly the variable the other two
    sides read (crewai resolves it through litellm, byLLM in nodes.jac), so one
    export keeps the comparison apples-to-apples. The default is byLLM's
    gpt-4o -- byLLM is the fidelity target. byLLM prefixes "openai/" for
    litellm's sake; the raw SDK wants the bare id, so a provider prefix is
    stripped rather than passed through.
  * No streaming. Neither of the other implementations streams, and the shared
    token meter (../mock_mailbox/token_meter.py) cannot read usage off a
    streamed response -- it would count the call under `streamed_calls` and
    the run would look cheaper than it was.

The client is constructed lazily and reachable only through this seam, so
importing the agent needs no API key, and the MockMailbox -- whose constructor
installs the token meter's patch on the openai SDK -- is always built before
the first request goes out.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Sequence

# Pinned identically on all three sides of the comparison, or the benchmark
# measures the model rather than the framework. gpt-4o is byLLM's default;
# crewai left unset falls back to gpt-4o-mini, which is why the shared knob
# matters (see eval/README.md, "Token cost").
DEFAULT_MODEL = "gpt-4o"

# byLLM runs its stages with byllm's default call params (jac.toml declares
# none): temperature 0.7, no max_tokens. Matched here rather than "improved" --
# a different temperature would be a hidden variable in the A/B comparison.
TEMPERATURE = 0.7

_client: Any | None = None


def build_client() -> Any:
    # Imported here rather than at module scope so that `import llm` works with
    # no `openai` installed and no key set -- the pipeline and the prompts can
    # be inspected without credentials.
    from openai import OpenAI

    # OPENAI_API_KEY is read natively by the SDK; deliberately not named here,
    # so an unset variable fails at the provider with the provider's own
    # message rather than being passed through as a literal.
    return OpenAI()


def get_client() -> Any:
    global _client
    if _client is None:
        _client = build_client()
    return _client


def set_client(client: Any | None) -> None:
    """Swap the client (tests use a stand-in). None falls back to OpenAI()."""
    global _client
    _client = client


def active_model_name() -> str:
    """The model this run will call: $OPENAI_MODEL_NAME or byLLM's default.

    'openai/gpt-4o' -> 'gpt-4o': byLLM writes the litellm provider prefix,
    which the raw SDK neither needs nor accepts.
    """
    name = os.environ.get("OPENAI_MODEL_NAME", "") or DEFAULT_MODEL
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return name


def complete(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Iterable[Mapping[str, Any]] | None = None,
    response_format: Mapping[str, Any] | None = None,
) -> Any:
    """One round trip. Returns the assistant message.

    No usage bookkeeping here: the shared token meter installed by MockMailbox
    patches the openai SDK's response path, so every call this function makes
    is counted identically to the other two implementations' calls, and no
    agent code counts its own tokens (eval/README.md, "Token cost").
    """
    payload: dict[str, Any] = {
        "model": active_model_name(),
        "messages": list(messages),
        "temperature": TEMPERATURE,
    }
    if tools:
        payload["tools"] = list(tools)
    if response_format is not None:
        payload["response_format"] = dict(response_format)
    response = get_client().chat.completions.create(**payload)
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


def tool_turn(call_id: str, content: str) -> dict[str, Any]:
    """One tool result, in the shape the API wants it answered."""
    return {"role": "tool", "tool_call_id": call_id, "content": content}


__all__ = [
    "DEFAULT_MODEL",
    "TEMPERATURE",
    "active_model_name",
    "assistant_turn",
    "build_client",
    "complete",
    "get_client",
    "set_client",
    "tool_turn",
]
