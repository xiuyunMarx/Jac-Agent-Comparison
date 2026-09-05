"""Adapter to swebench bridge：swe_entry.jac 从这里 import solve / active_model_name"""
import json;
import os;
import re;
import litellm;
import main;
import from nodes { llm, tool_log }

# token usage statistics
glob usage_log: list = [];
glob _orig_completion: any = litellm.completion;

def _counting_completion(*args: any, **kwargs: any) -> any {
    resp = _orig_completion(*args, **kwargs);
    u = getattr(resp, "usage", None);
    if u is not None {
        usage_log.append(u);
    }
    _trace_call(kwargs, resp);
    return resp;
}

# ---------------------------------------------------------------------------
# Per-call trace, when $CODEAGENT_TRACE names a file (the bridge points it at
# logs/<instance>/llm_trace.jsonl). One JSON line per model call: the request
# as litellm received it -- in full when the call opens a phase or routes,
# i.e. ends on a user message; otherwise just the newest message -- plus the
# usage and the reply. Same shape on all three arms, so traces diff directly.
# ---------------------------------------------------------------------------
glob _trace: dict = {"n": 0};
glob _KNOWN_KEYS: list[str] = ["model", "messages", "tools", "temperature", "response_format"];

def _as_str(x: any) -> str { return str(x); }

def _jsonable(x: any) -> any {
    if x is None or isinstance(x, (str, int, float, bool)) { return x; }
    if isinstance(x, dict) { return {str(k): _jsonable(v) for (k, v) in x.items()}; }
    if isinstance(x, (list, tuple)) { return [_jsonable(v) for v in x]; }
    if hasattr(x, "model_dump") {
        try { return _jsonable(x.model_dump(exclude_none=True)); } except Exception {}
    }
    if hasattr(x, "to_dict") {
        try { return _jsonable(x.to_dict()); } except Exception {}
    }
    if hasattr(x, "__dict__") { return _jsonable(vars(x)); }
    return str(x);
}

def _trace_call(payload: dict, resp: any) {
    _trace["n"] += 1;
    path = os.environ.get("CODEAGENT_TRACE", "");
    if not path { return; }
    messages: list = [_jsonable(m) for m in (payload.get("messages") or [])];
    last = messages[-1] if messages else {};
    opens = isinstance(last, dict) and last.get("role") == "user";
    tools: list = [_jsonable(t) for t in (payload.get("tools") or [])];
    choices = getattr(resp, "choices", None) or [];
    reply = _jsonable(choices[0].message) if choices else None;
    row = {
        "call": _trace["n"],
        "model": payload.get("model"),
        "n_messages": len(messages),
        "tools": [t["function"]["name"] for t in tools],
        "tool_schemas": tools if opens else None,
        "temperature": payload.get("temperature"),
        "response_format": _jsonable(payload.get("response_format")),
        "extra": sorted([k for k in payload.keys() if k not in _KNOWN_KEYS]),
        "usage": _jsonable(getattr(resp, "usage", None)),
        "messages": messages if opens else None,
        "last": None if opens else last,
        "reply": reply
    };
    with open(path, "a", encoding="utf-8") as f {
        f.write(json.dumps(row, `default=_as_str) + "\n");
    }
}

with entry {
    setattr(litellm, "completion", _counting_completion);
}


obj ToolCall {
    has name: str;
    has args: dict[str, str] = {};
}


obj RunResult {
    has objective: str;
    has answer: str = "";
    has steps: int = 0;
    has tool_calls: list[ToolCall] = [];
    has llm_calls: int = 0;
    has prompt_tokens: int = 0;
    has completion_tokens: int = 0;
    has cached_tokens: int = 0;
}


def _usage_int(u: any, key: str) -> int {
    try {
        if isinstance(u, dict) {
            return int(u.get(key, 0) or 0);
        }
        return int(getattr(u, key, 0) or 0);
    } except Exception {
        return 0;
    }
}


def _cached_tokens(u: any) -> int {
    n = _usage_int(u, "cache_read_input_tokens");
    if n {
        return n;
    }
    details = u.get("prompt_tokens_details") if isinstance(u, dict) else getattr(u, "prompt_tokens_details", None);
    if details is None {
        return 0;
    }
    return _usage_int(details, "cached_tokens");
}


def active_model_name() -> str {
    return str(llm.model_name);
}


def solve(objective: str, repo_root: str = "", max_steps: int = 10) -> RunResult {
    usage_log.clear();
    answer = main.solve(objective, repo_root or ".");
    history = list(usage_log);
    calls: list[ToolCall] = [
        ToolCall(name=str(c["name"]), args={"args": str(c["args"]), "ok": str(c["ok"])})
        for c in tool_log
    ];
    return RunResult(
        objective=objective,
        answer=answer,
        steps=len(re.findall(r"(?m)^### (Plan|Explore|Edit|Verify)\b", answer)),
        tool_calls=calls,
        llm_calls=len(history),
        prompt_tokens=sum([_usage_int(u, "prompt_tokens") for u in history]),
        completion_tokens=sum([_usage_int(u, "completion_tokens") for u in history]),
        cached_tokens=sum([_cached_tokens(u) for u in history])
    );
}
