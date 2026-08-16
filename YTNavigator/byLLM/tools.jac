"""LLM-facing tools for the byLLM YT-Navigator counterpart.

Thin Jac wrappers around retrieval.py: each call is logged (for the benchmark
result records) and described to the LLM with `sem`, mirroring the two tools
of the original LangGraph agent. Also hosts the litellm token/latency tracker
shared by every `by llm()` call in the process.
"""

import litellm;
import time;
import from retrieval { search_videos_impl, run_sql_impl, duration_seconds }

glob runtime_state: dict[str, str] = {"channel_id": ""};
glob tool_call_log: list[dict] = [];


def set_active_channel(channel_id: str) {
    runtime_state["channel_id"] = channel_id;
}


def reset_tool_log() {
    tool_call_log.clear();
}


def get_tool_calls() -> list[dict] {
    return list(tool_call_log);
}


def similarity_videos_search(query: str) -> str {
    tool_call_log.append({"name": "similarity_videos_search", "args": {"query": query}});
    try {
        return str(search_videos_impl(query, runtime_state["channel_id"]));
    } except Exception as e {
        return f"Search failed: {e}";
    }
}
sem similarity_videos_search = "Advanced semantic video search tool powered by vector embeddings. Use this tool to: find videos that match the semantic meaning of your query, not just exact keywords; discover relevant video content across different topics and contexts; retrieve video chunks that closely align with the intent of your search; get an idea about the channel's content. Ideal for complex information retrieval tasks over video transcripts. The tool may return some irrelevant results; provide the user with the most relevant ones.";
sem similarity_videos_search.query = "Natural-language search query to find similar transcript content.";


def execute_query(query: str) -> str {
    tool_call_log.append({"name": "execute_query", "args": {"query": query}});
    try {
        return str(run_sql_impl(query));
    } except Exception as e {
        return f"Query failed: {e}";
    }
}
sem execute_query = "Powerful SQL query execution tool (PostgreSQL syntax, SELECT only) for advanced data retrieval and analysis over the channel's videos: joining the app_video and app_videochunk tables, filtering and aggregating video metadata, counting videos, or retrieving specific subsets of data. app_video columns: id, title, thumbnail, published_at, channel_id. app_videochunk columns: id, video_id, start, end, text.";
sem execute_query.query = "A single SELECT statement against app_video and/or app_videochunk.";


def _usage_field(u: object, key: str) -> int {
    if isinstance(u, dict) {
        return int(u.get(key, 0) or 0);
    }
    return int(getattr(u, key, 0) or 0);
}


"""Aggregates per-call token usage and latency across every LiteLLM call in
the process. Registered once as a litellm success callback; byLLM dispatches
all `by llm()` calls through litellm, so this captures the whole agent run."""
obj TokenUsage {
    has calls: list[dict] = [];

    def track(kwargs: dict[str, object], response: object, start_time: object, end_time: object) {
        # Streaming calls carry usage on the reassembled response in kwargs.
        resp = kwargs.get("complete_streaming_response") or response;
        u = getattr(resp, "usage", None);
        prompt_tokens = None;
        completion_tokens = None;
        if u is not None {
            prompt_tokens = _usage_field(u, "prompt_tokens");
            completion_tokens = _usage_field(u, "completion_tokens");
        }
        self.calls.append({
            "model": getattr(resp, "model", None) or kwargs.get("model"),
            "latency_s": duration_seconds(start_time, end_time),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        });
    }

    def settle(timeout_s: float = 3.0, quiet_s: float = 0.3) {
        # LiteLLM runs success callbacks on a background thread pool, so the
        # records can lag the last call; wait until they stop moving.
        deadline = time.monotonic() + timeout_s;
        last = len(self.calls);
        quiet_since = time.monotonic();
        while time.monotonic() < deadline {
            time.sleep(0.05);
            if len(self.calls) != last {
                last = len(self.calls);
                quiet_since = time.monotonic();
            } elif time.monotonic() - quiet_since >= quiet_s {
                return;
            }
        }
    }
}

glob token_usage = TokenUsage();


def register_token_tracking() {
    if token_usage.track not in litellm.success_callback {
        litellm.success_callback.append(token_usage.track);
    }
}


def llm_call_count() -> int {
    return len(token_usage.calls);
}


def llm_calls_since(index: int) -> list[dict] {
    token_usage.settle();
    return list(token_usage.calls[index:]);
}
