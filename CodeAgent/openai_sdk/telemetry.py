"""Usage recording: what each round trip to the model cost, and what it carried.

Port of byLLM/logger/log_usage.jac, and the place where the absence of a
framework shows up as a deletion rather than as work.

byLLM records passively: it appends `TokenUsage.track` to
`litellm.success_callback`, litellm invokes it from a background pool, and
`settle()` then polls until the records stop moving -- because totals read
straight off the tracker can otherwise miss the final call, silently, making the
run look one call cheaper than it was. The four extractors on that side read an
untyped callback payload whose shape varies by provider, so every field is
optional-with-fallback.

Here the response is the return value of the call that produced it. Recording
happens on the calling thread, in `llm.complete`, with the request in one hand
and the response in the other -- so the callback registration, the settle loop,
and the guard against double registration all disappear. What is left is the
same record shape, because the eval harness and `llm_log.py` read it on both
sides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass
class LLMMessage:
    """One message in the prompt handed to the provider."""

    role: str = ""
    content: str = ""


@dataclass
class LLMToolCall:
    """One tool invocation the model asked for.

    `arguments` is the raw JSON string the provider emitted, kept verbatim so a
    malformed argument blob shows up in the log instead of being swallowed by a
    parse.
    """

    name: str = ""
    arguments: str = ""


@dataclass
class LLMCall:
    """One completed round trip to the model."""

    model: str = ""
    messages: list[LLMMessage] = field(default_factory=list)
    output: str = ""
    tools_offered: list[str] = field(default_factory=list)
    tools_called: list[LLMToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # A SUBSET of prompt_tokens, never an addition to it. See
    # extract_cached_tokens.
    cached_tokens: int = 0


def extract_messages(messages: Sequence[Mapping[str, Any]]) -> list[LLMMessage]:
    """The prompt as it was sent, flattened to role/content pairs.

    A tool-call turn carries its calls in `tool_calls` and no content at all;
    rendering the calls into the content line keeps the log readable, since
    otherwise the transcript shows an assistant turn that said nothing followed
    by tool results that answer nothing.
    """
    out: list[LLMMessage] = []
    for m in messages:
        role = str(m.get("role", "") or "")
        content = m.get("content")
        text = "" if content is None else str(content)
        calls = m.get("tool_calls") or []
        if calls and not text:
            named = ", ".join(
                str((c.get("function") or {}).get("name", "")) for c in calls
            )
            text = f"(tool calls: {named})"
        out.append(LLMMessage(role=role, content=text))
    return out


def extract_output(message: Any) -> str:
    """The assistant text of one completion; "" when the turn was tools-only."""
    return str(getattr(message, "content", "") or "")


def extract_tools_offered(tools: Iterable[Mapping[str, Any]] | None) -> list[str]:
    """Names of the tools the model was offered on this call.

    This is what identifies which phase a call belongs to -- Editing offers
    write_file, Verifying does not -- so it is worth recording even though the
    caller already knows.
    """
    names: list[str] = []
    for spec in tools or []:
        fn = spec.get("function") if isinstance(spec, Mapping) else None
        if isinstance(fn, Mapping):
            names.append(str(fn.get("name", "")))
    return names


def extract_tools_called(message: Any) -> list[LLMToolCall]:
    """The tools the model actually invoked on this call."""
    out: list[LLMToolCall] = []
    for call in getattr(message, "tool_calls", None) or []:
        fn = getattr(call, "function", None)
        out.append(
            LLMToolCall(
                name=str(getattr(fn, "name", "") or ""),
                arguments=str(getattr(fn, "arguments", "") or ""),
            )
        )
    return out


def extract_cached_tokens(usage: Any) -> int:
    """Prompt tokens the provider served from its cache.

    Nested a level below `usage` and optional at every step, so read
    defensively: a provider that omits it should cost this one figure, never
    the run. A SUBSET of prompt_tokens, never an addition to it -- the provider
    counts a cache hit in both and bills it at a discount.
    """
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    cached = (
        details.get("cached_tokens")
        if isinstance(details, dict)
        else getattr(details, "cached_tokens", None)
    )
    return int(cached or 0)


class TokenUsage:
    def __init__(self) -> None:
        self.calls: list[LLMCall] = []

    def track(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Iterable[Mapping[str, Any]] | None,
        response: Any,
    ) -> LLMCall:
        """Record one round trip. Returns the record, for the caller's use."""
        choices = getattr(response, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        usage = getattr(response, "usage", None)
        record = LLMCall(
            model=str(getattr(response, "model", None) or model or "?"),
            messages=extract_messages(messages),
            output=extract_output(message),
            tools_offered=extract_tools_offered(tools),
            tools_called=extract_tools_called(message),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cached_tokens=extract_cached_tokens(usage),
        )
        self.calls.append(record)
        return record

    def reset(self) -> None:
        self.calls.clear()

    def totals(self) -> dict[str, int]:
        prompt = 0
        completion = 0
        cached = 0
        for call in self.calls:
            prompt += call.prompt_tokens
            completion += call.completion_tokens
            cached += call.cached_tokens
        return {
            "llm_calls": len(self.calls),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cached_tokens": cached,
        }


__all__ = [
    "LLMCall",
    "LLMMessage",
    "LLMToolCall",
    "TokenUsage",
    "extract_messages",
    "extract_output",
    "extract_tools_called",
    "extract_tools_offered",
]
