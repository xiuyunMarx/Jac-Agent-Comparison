"""swebench_bridge 的适配层：swe_entry.jac 从这里 import solve / active_model_name"""
import re;
import litellm;
import main;
import from nodes { llm, tool_log }

# token 统计：byLLM 的非流式 ReAct 路径不记录 usage，telemetry 回调里也没有 token 字段，
# 所以直接在 litellm.completion 外面包一层同步计数。byLLM 每次都按模块属性调用它
# （model.impl.jac: litellm.completion(...)），替换后每一次模型调用都会经过这里。
glob usage_log: list = [];
glob _orig_completion: any = litellm.completion;

def _counting_completion(*args: any, **kwargs: any) -> any {
    resp = _orig_completion(*args, **kwargs);
    u = getattr(resp, "usage", None);
    if u is not None {
        usage_log.append(u);
    }
    return resp;
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
