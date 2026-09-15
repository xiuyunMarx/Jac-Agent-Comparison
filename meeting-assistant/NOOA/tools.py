"""Mock replacements for the CrewAI project's Trello and Slack helpers.

No network calls are made, so benchmark runs measure framework overhead
only. Every tool call is collected, and dump_outputs() writes them all to
one JSON file so a run can be evaluated by inspecting a single artifact.

A near-literal port of ../byLLM/tools.jac, as ../openai_sdk/tools.py is.
Same function names, same argument contracts, same printed lines, same
tool_outputs.json shape, so the eval harness in ../eval reads every side
identically.

Token accounting: NOOA routes every model call through one UnifiedLLM
client object, so nodes.py subclasses that client and feeds `token_usage`
from each response's usage block at the call site. Nothing runs on a
background thread, so (as on the openai_sdk side) the record is complete
before dump_outputs() runs and byLLM's settle() polling is not needed.
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
    """Aggregates token usage across every model call in the process.

    Same counters and snapshot() shape as byLLM's litellm success callback
    and the openai_sdk arm's call-site tracking. Fed by nodes.TracedClient
    from each LLMResponse.usage right after the call returns.
    """

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def track(self, usage: object) -> None:
        if usage is None:
            return
        self.prompt_tokens += _usage_field(usage, "prompt_tokens")
        self.completion_tokens += _usage_field(usage, "completion_tokens")
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
    """Nothing to hook. Kept so main.py mirrors the other entry points.

    The NOOA client subclass in nodes.py records usage at the call site, so
    there is no callback to register (byLLM appends a litellm callback,
    CrewAI patches the SDK transport).
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
