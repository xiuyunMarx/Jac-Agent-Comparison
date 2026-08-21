"""Mock replacements for the CrewAI project's Trello and Slack helpers.

No network calls are made, so benchmark runs measure framework overhead
only. Every tool call is collected, and dump_outputs() writes them all to
one JSON file so a run can be evaluated by inspecting a single artifact.

A near-literal port of ../byLLM/tools.jac (which is itself the mirror of
../CrewAI/src/meeting_assistant_flow/utils/). Same function names, same
argument contracts, same printed lines, same tool_outputs.json shape, so
the eval harness in ../eval reads all three sides identically.

The one deletion is byLLM's TokenUsage.settle(): litellm runs success
callbacks on a background thread pool, so that side must poll until the
counters stop moving or silently under-report the last call. Here usage
arrives on the response object and is recorded at the call site (nodes.py),
so the record is complete before dump_outputs() ever runs.
"""

import csv
import json

mock_trello_board: list[dict[str, str]] = []
mock_slack_messages: list[str] = []
mock_csv_rows: list[list[str]] = []


def create_trello_card(name: str, description: str) -> dict[str, str]:
    card = {"name": name, "desc": description}
    mock_trello_board.append(card)
    print(f"[mock trello] Task '{name}' successfully created in Trello.")
    return card


def send_message_to_channel(text: str) -> dict:
    mock_slack_messages.append(text)
    print(f"[mock slack] {text}")
    return {"ok": True, "text": text}


def save_tasks_to_csv(rows: list[tuple[str, str]], path: str = "new_tasks.csv") -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Description"])
        for row in rows:
            mock_csv_rows.append([row[0], row[1]])
            writer.writerow([row[0], row[1]])


def _usage_field(u: object, key: str) -> int:
    if isinstance(u, dict):
        return int(u.get(key, 0) or 0)
    return int(getattr(u, key, 0) or 0)


class TokenUsage:
    """Aggregates token usage across every OpenAI call in the process.

    Same counters and snapshot() shape as byLLM's litellm success callback
    and CrewAI's transport patch, but fed synchronously: nodes.py calls
    track(response) right after each chat.completions.create, and usage is
    already on the response, so nothing can lag or be missed.
    """

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def track(self, response: object) -> None:
        u = getattr(response, "usage", None)
        if u is None:
            return
        self.prompt_tokens += _usage_field(u, "prompt_tokens")
        self.completion_tokens += _usage_field(u, "completion_tokens")
        self.calls += 1

    def snapshot(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


token_usage = TokenUsage()


def register_token_tracking() -> None:
    """Nothing to hook. Kept so main.py mirrors the other two entry points.

    byLLM must append a litellm success callback and CrewAI must patch the
    SDK's shared transport, because in both cases the LLM call happens deep
    inside the framework. Here the call site is nodes.py and it records its
    own usage, so registration is genuinely a no-op rather than an omission.
    """


def dump_outputs(path: str = "tool_outputs.json") -> None:
    data = {
        "trello": mock_trello_board,
        "slack": mock_slack_messages,
        "csv": mock_csv_rows,
        "token_usage": token_usage.snapshot(),
    }
    with open(path, "w") as f:
        f.write(json.dumps(data, indent=2))
    print(f"[mock tools] Outputs collected in {path}")
