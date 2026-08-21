"""The two LLM-facing tools, with the call log the benchmark reads.

The counterpart of byLLM's `tools.jac`: thin wrappers around the shared
retrieval plumbing, each call logged as {"name", "args"} for the result
records, plus the explicit JSON Schema specs that `sem` strings (byLLM) and
`StructuredTool.from_function` + pydantic (LangGraph) were generating. The
descriptions are byLLM's `sem` texts verbatim, so all three sides advertise
the tools to the model in the same words.

The database plumbing itself -- pgvector semantic search, BM25, the SQL tool,
channel info -- is `../byLLM/retrieval.py`, loaded from the sibling rather
than copied: it is deliberately framework-neutral (stdlib + psycopg2 +
lazily-imported sentence-transformers), and one copy hitting one set of
tables is what keeps the comparison apples-to-apples. `eval/e2e.py` imports
the same file the same way for its retrieval sanity check.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_RETRIEVAL_PATH = Path(__file__).resolve().parent.parent / "byLLM" / "retrieval.py"


def _load_retrieval() -> Any:
    """Load the shared retrieval module from the byLLM sibling by file path."""
    if "ytnav_retrieval" in sys.modules:
        return sys.modules["ytnav_retrieval"]
    spec = importlib.util.spec_from_file_location("ytnav_retrieval", _RETRIEVAL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Shared retrieval module not found: {_RETRIEVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ytnav_retrieval"] = module
    spec.loader.exec_module(module)
    return module


retrieval = _load_retrieval()

# The channel every tool call is scoped to; the runner sets it once. The model
# never passes a channel id -- same contract as byLLM (the original app's tool
# took a channel_id argument the prompt had to supply; both counterparts
# inject it instead, see the README's divergences).
_runtime_state: dict[str, str] = {"channel_id": ""}

# {"name", "args"} per invocation, reset per question -- the result record's
# tool_calls field.
_tool_call_log: list[dict[str, Any]] = []


def set_active_channel(channel_id: str) -> None:
    _runtime_state["channel_id"] = channel_id


def reset_tool_log() -> None:
    _tool_call_log.clear()


def get_tool_calls() -> list[dict[str, Any]]:
    return list(_tool_call_log)


def similarity_videos_search(query: str) -> str:
    """Hybrid semantic + BM25 transcript search, formatted for the model."""
    _tool_call_log.append({"name": "similarity_videos_search", "args": {"query": query}})
    try:
        return str(retrieval.search_videos_impl(query, _runtime_state["channel_id"]))
    except Exception as e:  # noqa: BLE001 - the model reads the failure
        return f"Search failed: {e}"


def execute_query(query: str) -> str:
    """SELECT-only SQL over the channel's video/chunk tables."""
    _tool_call_log.append({"name": "execute_query", "args": {"query": query}})
    try:
        return str(retrieval.run_sql_impl(query))
    except Exception as e:  # noqa: BLE001
        return f"Query failed: {e}"


_HANDLERS = {
    "similarity_videos_search": similarity_videos_search,
    "execute_query": execute_query,
}

# What StructuredTool / `sem` were compiling to: the OpenAI function specs.
# Descriptions are byLLM's `sem` strings verbatim.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "similarity_videos_search",
            "description": (
                "Advanced semantic video search tool powered by vector embeddings. Use this tool to: "
                "find videos that match the semantic meaning of your query, not just exact keywords; "
                "discover relevant video content across different topics and contexts; retrieve video "
                "chunks that closely align with the intent of your search; get an idea about the "
                "channel's content. Ideal for complex information retrieval tasks over video "
                "transcripts. The tool may return some irrelevant results; provide the user with the "
                "most relevant ones."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query to find similar transcript content.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_query",
            "description": (
                "Powerful SQL query execution tool (PostgreSQL syntax, SELECT only) for advanced data "
                "retrieval and analysis over the channel's videos: joining the app_video and "
                "app_videochunk tables, filtering and aggregating video metadata, counting videos, or "
                "retrieving specific subsets of data. app_video columns: id, title, thumbnail, "
                "published_at, channel_id. app_videochunk columns: id, video_id, start, end, text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A single SELECT statement against app_video and/or app_videochunk.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def dispatch(name: str, raw_args: str) -> str:
    """Run one tool call. Never raises; every failure is a string the model reads."""
    handler = _HANDLERS.get(name)
    if handler is None:
        available = ", ".join(sorted(_HANDLERS))
        return f"Error: there is no tool named '{name}'. Available tools: {available}."
    try:
        args = json.loads(raw_args) if raw_args and raw_args.strip() else {}
    except ValueError as e:
        return f"Error: the arguments were not valid JSON ({e}). Re-issue the call with a JSON object."
    if not isinstance(args, dict):
        return f"Error: the arguments must be a JSON object of named parameters, not {type(args).__name__}."
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return "Error: this tool takes one required string parameter: 'query'."
    return handler(query)


__all__ = [
    "TOOL_SPECS",
    "dispatch",
    "execute_query",
    "get_tool_calls",
    "reset_tool_log",
    "retrieval",
    "set_active_channel",
    "similarity_videos_search",
]
