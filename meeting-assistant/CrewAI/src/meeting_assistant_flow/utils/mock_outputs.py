"""Collected mock tool outputs, dumped to one JSON file per run.

Mirrors the byLLM version's tools.jac collector so both frameworks can be
evaluated by inspecting a single, identically-shaped artifact.
"""

import functools
import json

mock_trello_board = []
mock_slack_messages = []
mock_csv_rows = []

_LLM_PATHS = ("/chat/completions", "/responses")


def _usage_field(usage, *keys):
    for key in keys:
        if isinstance(usage, dict):
            value = usage.get(key)
        else:
            value = getattr(usage, key, None)
        if value:
            return int(value)
    return 0


class TokenUsage:
    """Counts every OpenAI API call made by this process.

    CrewAI 1.x reaches the openai SDK through several independent layers: the
    native provider (chat.completions.create / beta parse / responses.create)
    and the instructor client that the output-Pydantic converter builds for
    itself, which bypasses CrewAI's event bus entirely. The one hook that sees
    all of them is the SDK's shared transport, SyncAPIClient.post /
    AsyncAPIClient.post. One post to an LLM endpoint == one call, matching how
    the byLLM side counts litellm completions.
    """

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def track(self, path, response):
        if not any(p in str(path) for p in _LLM_PATHS):
            return
        self.calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return  # e.g. a stream without include_usage: call counted, tokens unknown
        # chat API uses prompt/completion, responses API uses input/output
        self.prompt_tokens += _usage_field(usage, "prompt_tokens", "input_tokens")
        self.completion_tokens += _usage_field(usage, "completion_tokens", "output_tokens")

    def snapshot(self):
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


token_usage = TokenUsage()


def register_token_tracking():
    # Class-level patch, so it covers every client instance no matter which
    # layer constructed it (native provider, instructor, anything else).
    from openai._base_client import AsyncAPIClient, SyncAPIClient

    if getattr(SyncAPIClient.post, "_counts_tokens", False):
        return

    sync_post = SyncAPIClient.post
    async_post = AsyncAPIClient.post

    @functools.wraps(sync_post)
    def post(self, path, **kwargs):
        response = sync_post(self, path, **kwargs)
        token_usage.track(path, response)
        return response

    @functools.wraps(async_post)
    async def apost(self, path, **kwargs):
        response = await async_post(self, path, **kwargs)
        token_usage.track(path, response)
        return response

    post._counts_tokens = True
    apost._counts_tokens = True
    SyncAPIClient.post = post
    AsyncAPIClient.post = apost


def dump_outputs(path="tool_outputs.json"):
    data = {
        "trello": mock_trello_board,
        "slack": mock_slack_messages,
        "csv": mock_csv_rows,
        "token_usage": token_usage.snapshot(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[mock tools] Outputs collected in {path}")
