"""Framework-agnostic LLM token/cost meter for the mock-mailbox harness.

Both implementations in this repo end up talking to OpenAI through the `openai`
Python SDK -- CrewAI and byLLM both via litellm -- so metering one choke point
inside that SDK captures every LLM call regardless of framework, without
touching agent code.

The choke point is `SyncAPIClient._process_response` (and its async twin): every
response passes through it *after* the HTTP call, whether the caller used
`chat.completions.create`, `.with_raw_response.create` or `.parse`, so the meter
does not depend on which wrapper a framework happens to use (langchain-openai,
for one, goes through the raw-response and `.parse` wrappers).

Usage (normally automatic -- MockMailbox installs the meter on construction):

    from mock_mailbox.token_meter import install_token_meter
    meter = install_token_meter(reset=True)
    ...run the agent...
    meter.summary()   # -> {"llm_calls": 12, "prompt_tokens": ..., "cost_usd": ...}

Set EVAL_TOKEN_METER=0 to disable metering entirely.

Streaming responses are not metered: usage only arrives in the final SSE chunk
(and only with stream_options.include_usage), and none of the three agents
stream. `summary()["streamed_calls"]` counts any that slipped through so a
missing-token surprise is visible rather than silent.
"""

import json
import os
from pathlib import Path

# USD per 1M tokens: {model: (input, cached_input, output)}. Snapshot of OpenAI
# list prices taken 2026-08-10 -- override rather than trust blindly, via
# $EVAL_PRICING_FILE or mock_mailbox/pricing.json, in the same {model: [in,
# cached_in, out]} shape. Lookup is exact first, then longest-prefix, so dated
# snapshots ("gpt-4o-2024-08-06") and provider-prefixed ids ("openai/gpt-4o")
# resolve to their base model.
PRICING = {
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "o3": (2.00, 0.50, 8.00),
    "o3-mini": (1.10, 0.55, 4.40),
    "o4-mini": (1.10, 0.275, 4.40),
    "text-embedding-3-small": (0.02, 0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.13, 0.0),
}

PRICING_FILE = Path(__file__).resolve().parent / "pricing.json"


def _load_pricing_overrides():
    path = os.environ.get("EVAL_PRICING_FILE") or (
        str(PRICING_FILE) if PRICING_FILE.is_file() else ""
    )
    if not path:
        return
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:  # a bad price file must not break a run
        print(f"[token_meter] ignoring pricing file {path}: {exc}")
        return
    for model, price in data.items():
        if isinstance(price, dict):
            price = (price.get("input", 0.0),
                     price.get("cached_input", price.get("input", 0.0)),
                     price.get("output", 0.0))
        PRICING[model] = tuple(float(x) for x in price)


_load_pricing_overrides()


def normalize_model(model):
    """'openai/gpt-4o-2024-08-06' -> 'gpt-4o-2024-08-06' (drop provider prefix)."""
    model = (model or "").strip()
    if "/" in model:
        model = model.rsplit("/", 1)[-1]
    return model


def price_for(model):
    """(input, cached_input, output) USD per 1M tokens, or None if unpriced."""
    name = normalize_model(model)
    if name in PRICING:
        return PRICING[name]
    matches = [k for k in PRICING if name.startswith(k)]
    return PRICING[max(matches, key=len)] if matches else None


def cost_of(model, prompt_tokens=0, completion_tokens=0, cached_tokens=0):
    """USD cost of one call, or None when the model has no price entry.

    OpenAI counts cached tokens inside prompt_tokens, so they are billed at the
    cached rate and subtracted from the fresh-input count.
    """
    price = price_for(model)
    if price is None:
        return None
    p_in, p_cached, p_out = price
    fresh = max(prompt_tokens - cached_tokens, 0)
    return (fresh * p_in + cached_tokens * p_cached + completion_tokens * p_out) / 1e6


# -- usage extraction --------------------------------------------------------

def extract_usage(payload):
    """Pull a normalized usage dict out of a Chat Completions / Responses body.

    Returns None when the body carries no usage block (non-LLM endpoints,
    streamed responses, error payloads).
    """
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None

    # Chat Completions names them prompt/completion; Responses says input/output.
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    total = usage.get("total_tokens", 0) or (prompt + completion)
    in_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    out_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    return {
        "model": payload.get("model", ""),
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(total),
        "cached_prompt_tokens": int((in_details or {}).get("cached_tokens", 0) or 0),
        "reasoning_tokens": int((out_details or {}).get("reasoning_tokens", 0) or 0),
    }


TOKEN_FIELDS = ("prompt_tokens", "cached_prompt_tokens", "completion_tokens",
                "reasoning_tokens", "total_tokens")


class TokenMeter:
    """Accumulates per-call token usage and prices it."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = []
        self.streamed_calls = 0

    def record(self, usage, endpoint=""):
        """Record one normalized usage dict (as returned by extract_usage())."""
        model = usage.get("model", "")
        call = {"model": model, "endpoint": endpoint}
        call.update({f: int(usage.get(f, 0) or 0) for f in TOKEN_FIELDS})
        call["cost_usd"] = cost_of(
            model,
            prompt_tokens=call["prompt_tokens"],
            completion_tokens=call["completion_tokens"],
            cached_tokens=call["cached_prompt_tokens"],
        )
        self.calls.append(call)
        return call

    def summary(self, include_calls=True):
        """Aggregate totals, a per-model breakdown and (optionally) each call."""
        by_model, unpriced = {}, set()
        totals = {f: 0 for f in TOKEN_FIELDS}
        cost = 0.0
        for call in self.calls:
            model = call["model"] or "unknown"
            bucket = by_model.setdefault(
                model, {"calls": 0, **{f: 0 for f in TOKEN_FIELDS}, "cost_usd": 0.0})
            bucket["calls"] += 1
            for f in TOKEN_FIELDS:
                bucket[f] += call[f]
                totals[f] += call[f]
            if call["cost_usd"] is None:
                unpriced.add(model)
                bucket["cost_usd"] = None
            elif bucket["cost_usd"] is not None:
                bucket["cost_usd"] += call["cost_usd"]
            cost += call["cost_usd"] or 0.0

        for bucket in by_model.values():
            if bucket["cost_usd"] is not None:
                bucket["cost_usd"] = round(bucket["cost_usd"], 6)

        summary = {
            "llm_calls": len(self.calls),
            **totals,
            "cost_usd": round(cost, 6),
            "priced": not unpriced,
            "unpriced_models": sorted(unpriced),
            "streamed_calls": self.streamed_calls,
            "by_model": by_model,
        }
        if include_calls:
            summary["calls"] = self.calls
        return summary

    def format_line(self):
        s = self.summary(include_calls=False)
        cost = f"${s['cost_usd']:.4f}" if s["priced"] else f"${s['cost_usd']:.4f}+ (unpriced models)"
        return (f"{s['llm_calls']} LLM calls | "
                f"{s['prompt_tokens']:,} in + {s['completion_tokens']:,} out = "
                f"{s['total_tokens']:,} tokens | {cost}")


# -- openai SDK instrumentation ----------------------------------------------

_METER = TokenMeter()
_INSTALLED = False

# Endpoints whose bodies carry a usage block worth metering.
_METERED_PATHS = ("/chat/completions", "/completions", "/responses", "/embeddings")


def get_meter():
    """The process-wide meter every implementation reports through."""
    return _METER


def _endpoint_of(options):
    url = getattr(options, "url", "")
    return url if isinstance(url, str) else str(url)


def _meter_response(response, stream, options):
    """Record usage from one already-completed openai SDK response."""
    endpoint = _endpoint_of(options)
    if not any(endpoint.endswith(p) for p in _METERED_PATHS):
        return
    if stream:
        _METER.streamed_calls += 1
        return
    usage = extract_usage(response.json())
    if usage:
        _METER.record(usage, endpoint=endpoint)


def install(reset=False):
    """Patch the openai SDK so every completed response is metered.

    Idempotent, and a no-op (returning the meter unpatched) when openai is not
    importable or EVAL_TOKEN_METER=0. Never raises: metering must not be able
    to break an agent run.
    """
    global _INSTALLED
    if reset:
        _METER.reset()
    if _INSTALLED or os.environ.get("EVAL_TOKEN_METER", "1") == "0":
        return _METER
    try:
        from openai import _base_client
    except Exception as exc:
        print(f"[token_meter] openai SDK not instrumented ({exc}); "
              "token stats will be empty")
        return _METER

    def wrap(cls, is_async):
        original = cls._process_response

        if is_async:
            async def patched(self, **kwargs):
                result = await original(self, **kwargs)
                try:
                    _meter_response(kwargs.get("response"), kwargs.get("stream"),
                                    kwargs.get("options"))
                except Exception:
                    pass
                return result
        else:
            def patched(self, **kwargs):
                result = original(self, **kwargs)
                try:
                    _meter_response(kwargs.get("response"), kwargs.get("stream"),
                                    kwargs.get("options"))
                except Exception:
                    pass
                return result

        patched._token_meter_original = original
        cls._process_response = patched

    try:
        wrap(_base_client.SyncAPIClient, False)
        wrap(_base_client.AsyncAPIClient, True)
    except Exception as exc:
        print(f"[token_meter] could not instrument the openai SDK ({exc}); "
              "token stats will be empty")
        return _METER

    _INSTALLED = True
    return _METER


def uninstall():
    """Restore the original openai SDK methods (used by the tests)."""
    global _INSTALLED
    try:
        from openai import _base_client
    except Exception:
        _INSTALLED = False
        return
    for cls in (_base_client.SyncAPIClient, _base_client.AsyncAPIClient):
        original = getattr(cls._process_response, "_token_meter_original", None)
        if original is not None:
            cls._process_response = original
    _INSTALLED = False


# Friendlier name for callers outside this module.
install_token_meter = install
