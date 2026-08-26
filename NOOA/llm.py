"""The model seam: which model this run calls, and the one client it calls it on.

NOOA is model-agnostic through litellm, so a model is named rather than
configured: `get_llm_client("gpt-5")`, `get_llm_client("claude-sonnet-5")`,
`get_llm_client("ollama_chat/qwen3", api_base=...)`. That is the whole of the
provider story on this side -- there is no request to build, no tool schema to
send, and no loop to drive, because the runtime owns all three.

The client is constructed lazily and reachable only through this seam, so
importing the agent needs no API key: the capability boundary in phases.py can
be inspected, and the tools exercised, with no credentials anywhere.
"""

from __future__ import annotations

import os
from typing import Any

from telemetry import TokenUsage

# Pinned identically on all sides of the comparison, or the benchmark measures
# the model rather than the framework. Overridable per shell with
# CODEAGENT_MODEL, exactly as byLLM/jac.toml and openai_sdk/llm.py do it.
DEFAULT_MODEL = "gpt-5"

# gpt-5-shaped, and the same two numbers the other sides send.
#
# temperature 1: gpt-5 rejects every other value. max_tokens 16384: a reasoning
# model bills its hidden reasoning against the completion budget, and at 4096 it
# stops mid-cell -- which here truncates a *program*, so the run loses the turn
# to a SyntaxError rather than merely to shortened prose.
TEMPERATURE = float(os.environ.get("CODEAGENT_TEMPERATURE", "1.0"))
MAX_TOKENS = int(os.environ.get("CODEAGENT_MAX_TOKENS", "16384"))
# An OpenAI-compatible endpoint, for running the comparison against a local
# server. Empty means "let litellm route the model name", which is what the
# benchmark does.
API_BASE = os.environ.get("CODEAGENT_API_BASE", "")

_client: Any | None = None
_model: str | None = None

# Every call this process makes, in order. `solve` resets it per run.
token_usage = TokenUsage()


def active_model_name() -> str:
    """The model this run will call.

    Reads the environment rather than the client, so it answers before any
    client exists -- the SWE-bench shim asks for it while reporting a failed
    run, where constructing a provider would raise and lose the report.

    The name is passed through whole, provider prefix included. openai_sdk
    strips one because the raw SDK wants a bare id; this side is litellm-routed
    like byLLM's, so `ollama_chat/qwen3` and `openai/gpt-5` mean something here
    and losing the prefix would send the run to the wrong provider.
    """
    return _model or os.environ.get("CODEAGENT_MODEL", DEFAULT_MODEL)


def set_model(model: str | None) -> None:
    global _model, _client
    _model = model
    # The model is baked into the client at construction, so changing one
    # invalidates the other. Left stale, `set_model` would silently not take.
    _client = None


def build_client() -> Any:
    """One metered client for the model this run names.

    Imported here rather than at module scope so `import llm` works with no key
    set and no registry loaded, for the same reason openai_sdk defers `from
    openai import OpenAI`.
    """
    from nooa.unifiedllm.registry import get_llm_client

    overrides: dict[str, Any] = {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}
    if API_BASE:
        overrides["api_base"] = API_BASE
    return get_llm_client(active_model_name(), **overrides)


def get_client() -> Any:
    """The client for this run, metered.

    The meter is attached here rather than in `build_client`, so a client handed
    in through `set_client` is counted on exactly the same terms as one this
    module built. `attach` is idempotent, so asking twice does not double-count.
    """
    global _client
    if _client is None:
        _client = build_client()
    return token_usage.attach(_client)


def set_client(client: Any | None) -> None:
    """Swap the client. Pass None to fall back to the configured provider.

    A test passes `nooa.unifiedllm.FakeLLMClient` here and the whole agent runs
    with no network at all -- the phase methods, the REPL, the tools and the
    ledger, on scripted cells.
    """
    global _client
    _client = client


__all__ = [
    "API_BASE",
    "DEFAULT_MODEL",
    "MAX_TOKENS",
    "TEMPERATURE",
    "active_model_name",
    "build_client",
    "get_client",
    "set_client",
    "set_model",
    "token_usage",
]
