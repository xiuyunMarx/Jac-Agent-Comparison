"""Writes the recorded LLM calls out as one readable file per call.

Port of byLLM/logger/log_LLM_history.jac. The LangGraph side has no equivalent;
this is the one thing byLLM does that had to be written rather than dropped.

Holds no state: everything it writes comes from the records `telemetry.TokenUsage`
accumulated during the run. A call is the unit because a call is what gets
read -- a single transcript of a twenty-call run is a file nobody scrolls
through, while `call-014.txt` is one prompt, one answer, and the tools that went
with it.

Nothing here goes to stdout, which the eval harness reads for the agent's
answer; the only thing that reaches the terminal is a one-line note saying where
the files landed.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Sequence

from telemetry import LLMCall, LLMToolCall, TokenUsage

# Written under the working directory, which is where the run was launched from
# and not the repository being edited -- nothing here can end up in a patch
# taken from the workspace.
DEFAULT_LOG_DIR = "llm_calls"

RULE = "=" * 78
SUBRULE = "-" * 78

# What a backslash can legally stand for inside a JSON string. A backslash
# followed by anything else is left exactly as it was found.
ESCAPES: dict[str, str] = {
    "n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f",
    '"': '"', "'": "'", "\\": "\\", "/": "/",
}


def unescape(text: str) -> str:
    """Turn escape sequences back into the characters they stand for.

    Walks the string a character at a time rather than calling
    `str.replace("\\n", "\n")`, which cannot tell a line break from the two
    characters that follow the backslash in an escaped backslash.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in ESCAPES:
            out.append(ESCAPES[text[i + 1]])
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def readable(text: str) -> str:
    """`text` with its escapes resolved -- but only if escapes are all it has.

    Text that already carries a real line break is handed back untouched. It is
    not an escaped blob, and rewriting the `\\n` inside a code sample the model
    quoted would corrupt the very thing the log exists to show.
    """
    if "\n" in text or "\\n" not in text:
        return text
    return unescape(text)


def indent(text: str, pad: str) -> list[str]:
    """`text` as log lines, each one indented by `pad`. Empty text yields nothing."""
    return [pad + ln for ln in readable(text).splitlines()]


def render_tool_call(index: int, tc: LLMToolCall) -> list[str]:
    """One tool call: its name, then every argument as its own labelled block.

    Arguments arrive as the raw JSON string the provider emitted, which puts the
    entire body of a written file on one line with its newlines spelled out.
    Decoding the JSON is what unspells them, and it is exact where a text
    substitution would only be a guess. A blob that will not parse is printed as
    it came, with its escapes resolved as far as they can be.
    """
    lines: list[str] = ["", f"[{index}] {tc.name}"]
    args: Any = {}
    parsed = True
    if tc.arguments:
        try:
            args = json.loads(tc.arguments)
        except Exception:  # noqa: BLE001 - a provider may emit anything
            parsed = False
    # Only a JSON object has arguments to take apart; anything else is shown as
    # it arrived.
    if not parsed or not isinstance(args, dict):
        lines.extend(indent(tc.arguments, "      "))
        return lines
    if not args:
        lines.append("      (no arguments)")
    for key, value in args.items():
        text = value if isinstance(value, str) else json.dumps(value, indent=2)
        if len(text.splitlines()) > 1:
            lines.append(f"      {key}:")
            lines.extend(indent(text, "        "))
        else:
            lines.append(f"      {key}: {text}")
    return lines


def render_call(index: int, total: int, call: LLMCall) -> str:
    """One call as the whole text of its file: header, prompt, output, tool calls."""
    offered = ", ".join(call.tools_offered) if call.tools_offered else "(none)"
    lines: list[str] = [
        RULE,
        f"call {index} of {total}",
        f"model          {call.model}",
        f"tokens         {call.prompt_tokens} prompt + {call.completion_tokens} completion",
        f"tools offered  {offered}",
        RULE,
        "",
        SUBRULE,
        f"PROMPT -- {len(call.messages)} message(s)",
        SUBRULE,
    ]
    if not call.messages:
        lines.append("")
        lines.append("    (not captured)")
    for i, m in enumerate(call.messages):
        lines.append("")
        lines.append(f"[{i + 1}] {m.role}")
        body = indent(m.content, "    ")
        lines.extend(body if body else ["    (empty)"])

    lines.extend(["", SUBRULE, "OUTPUT", SUBRULE, ""])
    out = indent(call.output, "    ")
    lines.extend(out if out else ["    (no text -- tool-call turn)"])

    lines.extend(["", SUBRULE, f"TOOL CALLS -- {len(call.tools_called)}", SUBRULE])
    if not call.tools_called:
        lines.append("")
        lines.append("    (none)")
    for i, tc in enumerate(call.tools_called):
        lines.extend(render_tool_call(i + 1, tc))
    return "\n".join(lines) + "\n"


def render_index(
    calls: Sequence[LLMCall], totals: dict[str, int], names: Sequence[str]
) -> str:
    """The one-page view: every call on a line, and what the run cost in total."""
    lines: list[str] = [RULE, f"LLM CALL LOG -- {len(calls)} call(s)", RULE, ""]
    for i, call in enumerate(calls):
        called = ", ".join(tc.name for tc in call.tools_called) if call.tools_called else "-"
        lines.append(
            f"{names[i]}  {call.model}  "
            f"{call.prompt_tokens}+{call.completion_tokens} tok  "
            f"tools: {called}"
        )
    lines.extend([
        "",
        RULE,
        f"TOTAL {totals['llm_calls']} call(s), {totals['prompt_tokens']} prompt "
        f"+ {totals['completion_tokens']} completion tokens",
        RULE,
    ])
    return "\n".join(lines) + "\n"


def log_llm_calls(folder: str = "", usage: TokenUsage | None = None) -> str:
    """Write every call the run made to `folder`, one file each, plus an index.

    Destination is `folder`, else $CODEAGENT_LLM_LOG, else `llm_calls/`. Returns
    the directory written, or "" if it could not be written -- in which case the
    whole transcript goes to stderr instead, since losing the log outright is the
    worse outcome.

    `call-*.txt` files already in the directory are cleared first, and only
    those: a short run would otherwise inherit the tail of a longer one, where it
    reads as part of the run just finished.
    """
    # byLLM settles here as well as in `solve`, because litellm's callbacks may
    # still be in flight. Nothing is in flight here -- `llm.complete` records
    # before it returns -- so the records are simply read.
    if usage is None:
        from llm import token_usage as usage_default

        usage = usage_default
    calls = usage.calls
    totals = usage.totals()
    names = [f"call-{(i + 1):03d}.txt" for i in range(len(calls))]
    dest = folder or os.environ.get("CODEAGENT_LLM_LOG", "") or DEFAULT_LOG_DIR

    try:
        os.makedirs(dest, exist_ok=True)
        for stale in os.listdir(dest):
            if stale.startswith("call-") and stale.endswith(".txt"):
                os.remove(os.path.join(dest, stale))
        for i, call in enumerate(calls):
            with open(os.path.join(dest, names[i]), "w", encoding="utf-8") as fh:
                fh.write(render_call(i + 1, len(calls), call))
        with open(os.path.join(dest, "index.txt"), "w", encoding="utf-8") as fh:
            fh.write(render_index(calls, totals, names))
    except OSError as e:
        sys.stderr.write(f"[llm-log] cannot write {dest}: {e}\n")
        sys.stderr.write(render_index(calls, totals, names))
        for i, call in enumerate(calls):
            sys.stderr.write(render_call(i + 1, len(calls), call))
        return ""
    sys.stderr.write(f"[llm-log] {len(calls)} call(s) written to {dest}{os.sep}\n")
    return dest


__all__ = [
    "DEFAULT_LOG_DIR",
    "indent",
    "log_llm_calls",
    "readable",
    "render_call",
    "render_index",
    "render_tool_call",
    "unescape",
]
