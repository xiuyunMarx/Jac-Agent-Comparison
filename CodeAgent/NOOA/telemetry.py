"""Usage recording: what each round trip to the model cost.

The comparison's other three sides each pay for this differently. byLLM appends
`TokenUsage.track` to `litellm.success_callback`, litellm runs those callbacks
on a background pool, and `settle()` then polls until the records stop moving --
because totals read straight off the tracker can otherwise miss the final call,
silently, making a run look one call cheaper than it was. LangGraph reads
`usage_metadata` off the message. openai_sdk reads `response.usage` at the call
site.

Here the response is not in reach of the agent code at all: NOOA owns the loop,
so nothing in orchestrator.py ever sees an LLMResponse. What is in reach is the
client, and every call goes through exactly one method on it. So the meter wraps
that method: `attach()` replaces `acall`/`call` on the client instance with a
wrapper that records the response on the way back and returns it untouched.

That keeps the byLLM property this file cares about -- accounting that cannot
miss a call -- without the callback registration or the settle loop, because the
recording happens on the calling task, in front of the caller, before the value
is handed back.

The record shape is the one `swebench_bridge` reads on every side:
llm_calls / prompt_tokens / completion_tokens / cached_tokens.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMCall:
    """One completed round trip to the model."""

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # A SUBSET of prompt_tokens, never an addition to it: a provider counts a
    # cache hit in both and bills it at a discount.
    cached_tokens: int = 0
    finish_reason: str = ""
    tools_called: list[str] = field(default_factory=list)


def extract_cached_tokens(usage: Any) -> int:
    """Prompt tokens the provider served from its cache.

    Nested a level below `usage` and optional at every step, so read
    defensively: a provider that omits it should cost this one figure, never the
    run. Two spellings, because two providers: OpenAI reports
    `prompt_tokens_details.cached_tokens`, Anthropic `cache_read_input_tokens`.
    unifiedllm hands the usage over as whatever `model_dump()` produced, so the
    nested level may be a dict or an object.
    """
    if usage is None:
        return 0
    get = usage.get if isinstance(usage, dict) else lambda k, d=None: getattr(usage, k, d)
    flat = get("cache_read_input_tokens", 0)
    if flat:
        return int(flat or 0)
    details = get("prompt_tokens_details", None)
    if details is None:
        return 0
    cached = (
        details.get("cached_tokens")
        if isinstance(details, dict)
        else getattr(details, "cached_tokens", None)
    )
    return int(cached or 0)


class TokenUsage:
    """Every call this process made, in order."""

    def __init__(self) -> None:
        self.calls: list[LLMCall] = []

    def observe(self, model: str, response: Any) -> LLMCall:
        """Record one round trip. Returns the record, for the caller's use."""
        usage = getattr(response, "usage", None)
        get = (
            usage.get
            if isinstance(usage, dict)
            else (lambda k, d=0: getattr(usage, k, d) if usage is not None else d)
        )
        record = LLMCall(
            model=model or "?",
            prompt_tokens=int(get("prompt_tokens", 0) or 0),
            completion_tokens=int(get("completion_tokens", 0) or 0),
            cached_tokens=extract_cached_tokens(usage),
            finish_reason=str(getattr(response, "finish_reason", "") or ""),
            tools_called=[
                str(getattr(c, "name", "") or "")
                for c in (getattr(response, "tool_calls", None) or [])
            ],
        )
        self.calls.append(record)
        return record

    def attach(self, client: Any) -> Any:
        """Meter `client` in place and return it.

        Wraps the two entry points every generation goes through. Both are
        replaced as *instance* attributes, so nothing else that holds this
        class is affected and a second attach on the same object is a no-op
        rather than a double count.
        """
        if getattr(client, "_metered_by", None) is self:
            return client
        model = str(getattr(client, "model", "") or "")

        for name in ("acall", "call"):
            inner = getattr(client, name, None)
            if inner is None:
                continue
            setattr(client, name, self._wrap(inner, model, is_async=name == "acall"))
        client._metered_by = self
        return client

    def _wrap(self, inner: Any, model: str, *, is_async: bool) -> Any:
        if is_async:

            @functools.wraps(inner)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                response = await inner(*args, **kwargs)
                self.observe(model, response)
                return response

            return awrapper

        @functools.wraps(inner)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            response = inner(*args, **kwargs)
            self.observe(model, response)
            return response

        return wrapper

    def reset(self) -> None:
        self.calls.clear()

    def totals(self) -> dict[str, int]:
        return {
            "llm_calls": len(self.calls),
            "prompt_tokens": sum(c.prompt_tokens for c in self.calls),
            "completion_tokens": sum(c.completion_tokens for c in self.calls),
            "cached_tokens": sum(c.cached_tokens for c in self.calls),
        }


__all__ = ["LLMCall", "TokenUsage", "extract_cached_tokens"]
