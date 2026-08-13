"""Usage recording: what each round trip to the LLM backend cost, and what it
carried.

Recording is passive. litellm invokes `TokenUsage.track` from its
success-callback pool, so nothing on the agent's own path -- no phase, no tool,
no walker ability -- has to remember to log. The records accumulated here are
the only input to `logger.log_LLM_history`, and the source of the token counts
`solve` reports.

Mirrors the TokenUsage tracker in the sibling byLLM implementations so the eval
harness reads the same shape on both sides of the comparison.
"""

import time;
import litellm;


"""One message in the prompt handed to the provider."""
obj LLMMessage {
    has role: str = "";
    has content: str = "";
}

"""One tool invocation the model asked for. `arguments` is the raw JSON string
the provider emitted, kept verbatim so a malformed argument blob shows up in
the log instead of being swallowed by a parse."""
obj LLMToolCall {
    has name: str = "";
    has arguments: str = "";
}

"""One completed round trip to the LLM backend."""
obj LLMCall {
    has model: str = "";
    has messages: list[LLMMessage] = [];
    has output: str = "";
    has tools_offered: list[str] = [];
    has tools_called: list[LLMToolCall] = [];
    has prompt_tokens: int = 0;
    has completion_tokens: int = 0;
}


# The four extractors below read litellm's untyped callback payload, so their
# locals are `any` by necessity -- the shape varies by provider and by whether
# the call streamed. Every field is optional-with-fallback: a provider that
# omits one costs that line of the log, never the run.

"""The prompt handed to the provider, flattened to role/content pairs."""
def extract_messages(kwargs: dict[str, object]) -> list[LLMMessage] {
    raw: any = kwargs.get("messages") or [];
    out: list[LLMMessage] = [];
    for m in raw {
        if isinstance(m, dict) {
            out.append(LLMMessage(
                role=str(m.get("role", "")),
                content=str(m.get("content", "") or "")
            ));
        } else {
            out.append(LLMMessage(
                role=str(getattr(m, "role", "") or ""),
                content=str(getattr(m, "content", "") or "")
            ));
        }
    }
    return out;
}


"""The assistant text of one completion; "" when the turn was tools-only."""
def extract_output(resp: object) -> str {
    choices: any = getattr(resp, "choices", None);
    if not choices {
        return "";
    }
    msg: any = getattr(choices[0], "message", None);
    if msg is None {
        return "";
    }
    return str(getattr(msg, "content", "") or "");
}


"""Names of the tools the provider was offered on this call.

byLLM passes the phase's toolset straight through, so this is what identifies
which phase a call belongs to -- Editing offers write_file, Verifying does not.
"""
def extract_tools_offered(kwargs: dict[str, object]) -> list[str] {
    opts: any = kwargs.get("optional_params");
    nested: any = opts.get("tools") if isinstance(opts, dict) else None;
    specs: any = kwargs.get("tools") or nested or [];
    names: list[str] = [];
    for t in specs {
        fn: any = t.get("function") if isinstance(t, dict) else getattr(t, "function", None);
        if isinstance(fn, dict) {
            names.append(str(fn.get("name", "")));
        } elif fn is not None {
            names.append(str(getattr(fn, "name", "") or ""));
        }
    }
    return names;
}


"""The tools the model actually invoked on this call."""
def extract_tools_called(resp: object) -> list[LLMToolCall] {
    out: list[LLMToolCall] = [];
    choices: any = getattr(resp, "choices", None);
    if not choices {
        return out;
    }
    msg: any = getattr(choices[0], "message", None);
    if msg is None {
        return out;
    }
    calls: any = getattr(msg, "tool_calls", None);
    if not calls {
        return out;
    }
    for c in calls {
        fn: any = getattr(c, "function", None);
        out.append(LLMToolCall(
            name=str(getattr(fn, "name", "") or ""),
            arguments=str(getattr(fn, "arguments", "") or "")
        ));
    }
    return out;
}


obj TokenUsage {
    has calls: list[LLMCall] = [];

    def track(kwargs: dict[str, object], response: object, _start: object, _end: object) {
        resp = kwargs.get("complete_streaming_response") or response;
        usage = getattr(resp, "usage", None);
        prompt_tokens = 0;
        completion_tokens = 0;
        if usage is not None {
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0);
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0);
        }
        self.calls.append(LLMCall(
            model=str(getattr(resp, "model", None) or kwargs.get("model") or "?"),
            messages=extract_messages(kwargs),
            output=extract_output(resp),
            tools_offered=extract_tools_offered(kwargs),
            tools_called=extract_tools_called(resp),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        ));
    }

    def settle(timeout_s: float = 3.0, quiet_s: float = 0.3) {
        # litellm runs success callbacks on a background pool, so the records
        # can lag the last call; wait until they stop moving.
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

    def totals() -> dict[str, int] {
        prompt = 0;
        completion = 0;
        for call in self.calls {
            prompt += call.prompt_tokens;
            completion += call.completion_tokens;
        }
        return {
            "llm_calls": len(self.calls),
            "prompt_tokens": prompt,
            "completion_tokens": completion
        };
    }
}

glob token_usage: TokenUsage = TokenUsage();


"""Start recording. Guarded rather than unconditional: `solve` calls this once
per run, and a second registration would have litellm invoke the callback twice
per call, doubling every count."""
def register_token_tracking() {
    if token_usage.track not in litellm.success_callback {
        litellm.success_callback.append(token_usage.track);
    }
}


"""What the run cost, once the in-flight callbacks have landed.

Settling is the reason this wrapper exists. Totals read straight off the
tracker can miss the final call, and the miss is silent -- the run simply looks
one call cheaper than it was.
"""
def usage_totals() -> dict[str, int] {
    token_usage.settle();
    return token_usage.totals();
}
