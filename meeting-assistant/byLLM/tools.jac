"""Mock replacements for the CrewAI project's Trello and Slack helpers.
No network calls are made, so benchmark runs measure framework overhead
only. Every tool call is collected, and dump_outputs() writes them all to
one JSON file so a run can be evaluated by inspecting a single artifact."""

import litellm;
import time;

glob mock_trello_board: list[dict[str, str]] = [];
glob mock_slack_messages: list[str] = [];
glob mock_csv_rows: list[list[str]] = [];

def create_trello_card(name: str, description: str) -> dict[str, str] {
    card = {"name": name, "desc": description};
    mock_trello_board.append(card);
    print(f"[mock trello] Task '{name}' successfully created in Trello.");
    return card;
}

def send_message_to_channel(text: str) -> dict[str, str | bool] {
    mock_slack_messages.append(text);
    print(f"[mock slack] {text}");
    return {"ok": True, "text": text};
}

def save_tasks_to_csv(rows: list[tuple[str, str]], path: str = "new_tasks.csv") {
    import csv;
    f = open(path, "w", newline="");
    writer = csv.writer(f);
    writer.writerow(["Name", "Description"]);
    for row in rows {
        mock_csv_rows.append([row[0], row[1]]);
        writer.writerow([row[0], row[1]]);
    }
    f.close();
}

def _usage_field(u: object, key: str) -> int {
    if isinstance(u, dict) {
        return int(u.get(key, 0) or 0);
    }
    return int(getattr(u, key, 0) or 0);
}

"""Aggregates token usage across every LiteLLM call in the process.
Registered once as a litellm success callback; byLLM dispatches all
`by llm()` calls through litellm, so this captures the whole agent run."""
obj TokenUsage {
    has prompt_tokens: int = 0;
    has completion_tokens: int = 0;
    has calls: int = 0;

    def track(kwargs: dict[str, object], response: object, _start_time: object, _end_time: object) {
        # Streaming calls carry usage on the reassembled response in kwargs.
        resp = kwargs.get("complete_streaming_response") or response;
        u = getattr(resp, "usage", None);
        if u is None {
            return;
        }
        self.prompt_tokens += _usage_field(u, "prompt_tokens");
        self.completion_tokens += _usage_field(u, "completion_tokens");
        self.calls += 1;
    }

    def snapshot -> dict[str, int] {
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens
        };
    }

    def settle(timeout_s: float = 3.0, quiet_s: float = 0.3) {
        # LiteLLM runs success callbacks on a background thread pool, so the
        # counters can lag the last call; wait until they stop moving.
        deadline = time.monotonic() + timeout_s;
        last = self.calls;
        quiet_since = time.monotonic();
        while time.monotonic() < deadline {
            time.sleep(0.05);
            if self.calls != last {
                last = self.calls;
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

def dump_outputs(path: str = "tool_outputs.json") {
    import json;
    token_usage.settle();
    data = {
        "trello": mock_trello_board,
        "slack": mock_slack_messages,
        "csv": mock_csv_rows,
        "token_usage": token_usage.snapshot()
    };
    f = open(path, "w");
    f.write(json.dumps(data, indent=2));
    f.close();
    print(f"[mock tools] Outputs collected in {path}");
}
