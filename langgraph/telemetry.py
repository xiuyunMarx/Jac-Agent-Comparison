"""Token accounting.

Mirrors the `TokenUsage` tracker in byLLM/orchestrator.jac so the eval harness
reads the same shape on both sides.

The byLLM version has to hook `litellm.success_callback` and then `settle()` --
poll until the records stop moving -- because litellm runs its success callbacks
on a background pool and they lag the last call. LangChain returns usage on the
response object itself, synchronously, so both the callback registration and the
settle loop disappear here. That deletion is a comparison datum, not an
oversight.
"""

from __future__ import annotations

from typing import Any


class TokenUsage:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def track(self, response: Any) -> None:
        """Record one LLM response. Accepts anything with usage_metadata."""
        if response is None:
            return
        prompt_tokens = 0
        completion_tokens = 0
        usage = getattr(response, "usage_metadata", None) or {}
        if usage:
            prompt_tokens = int(usage.get("input_tokens", 0) or 0)
            completion_tokens = int(usage.get("output_tokens", 0) or 0)
        else:
            # Older providers put it here instead; a fake model in the tests has
            # neither, and contributes a call with zero tokens.
            meta = getattr(response, "response_metadata", None) or {}
            legacy = meta.get("token_usage") or {}
            prompt_tokens = int(legacy.get("prompt_tokens", 0) or 0)
            completion_tokens = int(legacy.get("completion_tokens", 0) or 0)
        meta = getattr(response, "response_metadata", None) or {}
        self.calls.append({
            "model": meta.get("model_name") or meta.get("model"),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })

    def reset(self) -> None:
        self.calls.clear()

    def totals(self) -> dict[str, int]:
        prompt = 0
        completion = 0
        for call in self.calls:
            prompt += int(call.get("prompt_tokens", 0) or 0)
            completion += int(call.get("completion_tokens", 0) or 0)
        return {
            "llm_calls": len(self.calls),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
        }
