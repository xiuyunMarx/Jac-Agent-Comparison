"""Shared foundation for the coding agent's LLM-facing tools.

Every tool here returns `str`. LangChain stringifies anything that is not
already a str before it becomes a ToolMessage, so a `list[str]` would reach the
model as a Python repr and an empty one as the literal "[]" -- indistinguishable
from a broken tool.

The string sub-contract, used uniformly by every tool:

  read success      the content, led by one context line
                    "# tools/edit.py - lines 1-54 of 54"
  mutation success  starts with "OK: ", naming what changed and by how much
  recoverable error starts with "Error: " and always names the next action
  policy refusal    starts with "BLOCKED: " -- a distinct prefix, because
                    "Error" means "try again differently" while "BLOCKED"
                    means "this class of action is unavailable, stop trying"

Tools never raise, never print, and never return "". A tool that raises reaches
the model as a LangChain error string with an absolute-path traceback, which
pollutes the benchmark log; and an empty return is the classic ReAct-loop
trigger.

This module is a 1:1 port of byLLM/nodes/common.jac. The two sides of the
comparison must present an identical action space, so behaviour here is fixed by
that file rather than by Python convention.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Directories never worth walking in a source repo.
SKIP_DIRS: set[str] = {
    ".git", ".jac", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build",
}

# Subtrees no tool may ever write into. `.git/config` accepts `[diff] external`,
# which would turn a later `git diff` into arbitrary code execution.
NO_WRITE_DIRS: set[str] = {".git", ".jac"}

MAX_FILE_BYTES: int = 2 * 1024 * 1024
MAX_LINE_CHARS: int = 400
MAX_TOOL_CHARS: int = 20000
MAX_GREP_MATCHES: int = 100
MAX_LIST_ENTRIES: int = 300


@dataclass
class ToolCall:
    """One entry in the ordered record of tool calls, read by the eval harness.

    Mirrors the `ToolCall` object carried by byLLM/nodes/common.jac, so
    `RunResult.tool_calls` has the same shape on both sides.
    """

    name: str
    args: dict[str, str] = field(default_factory=dict)


tool_call_log: list[ToolCall] = []


def log_tool_call(name: str, args: dict[str, str]) -> None:
    tool_call_log.append(ToolCall(name=name, args=args))


def get_tool_calls() -> list[ToolCall]:
    return list(tool_call_log)


def reset_tool_log() -> None:
    tool_call_log.clear()


def safe_path(repo_root: str, rel_path: str) -> str | None:
    """Resolve `rel_path` under `repo_root`, or None if it escapes."""
    base = os.path.realpath(repo_root or os.getcwd())
    joined = rel_path if os.path.isabs(rel_path) else os.path.join(base, rel_path)
    # realpath, not abspath: abspath normalizes `..` textually and would let a
    # symlink inside the repo pointing at /etc through. realpath also resolves
    # the existing prefix of a not-yet-created path, so new-file writes confine.
    cand = os.path.realpath(joined)
    if cand == base:
        return cand
    try:
        # commonpath compares path segments; startswith would accept a sibling
        # directory such as `<root>-evil`.
        if os.path.commonpath([base, cand]) != base:
            return None
    except ValueError:
        # Raised on mixed absolute/relative input, or mixed drives on Windows.
        return None
    return cand


def is_write_blocked(repo_root: str, abs_path: str) -> bool:
    """True if `abs_path` lies in a subtree no tool may write to."""
    base = os.path.realpath(repo_root or os.getcwd())
    rel = os.path.relpath(abs_path, base)
    first = rel.split(os.sep)[0]
    return first in NO_WRITE_DIRS


def rel_to(repo_root: str, abs_path: str) -> str:
    """Repo-relative form of `abs_path`, for output the model reads."""
    base = os.path.realpath(repo_root or os.getcwd())
    try:
        return os.path.relpath(abs_path, base)
    except ValueError:
        return abs_path


def refuse_path(path: str) -> str:
    return (
        f"BLOCKED: path '{path}' resolves outside the repository root and was "
        "refused. Use a path relative to the repository root."
    )


def clip(text: str, limit: int) -> str:
    """Keep the head of `text`; mark what was dropped."""
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return text[:limit] + f"\n... [truncated, {dropped} more characters]"


def clip_ends(text: str, limit: int) -> str:
    """Keep the head AND tail of `text`.

    Command output puts the verdict at the end and the traceback at the start,
    so head-only truncation systematically discards the line that matters.
    """
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - (2 * half)
    return text[:half] + f"\n... [{omitted} characters omitted] ...\n" + text[-half:]


def scrub(text: str, repo_root: str) -> str:
    """Strip machine-specific detail from captured output.

    Absolute paths, the home directory and wall-clock durations all differ
    between runs and between machines; leaving them in would make the A/B
    comparison sensitive to where the benchmark happens to be checked out.
    """
    base = os.path.realpath(repo_root or os.getcwd())
    out = text.replace(base + os.sep, "").replace(base, ".")
    home = os.path.expanduser("~")
    if home and home != "/":
        out = out.replace(home, "~")
    out = re.sub(r"\b\d+\.\d+s\b", "X.XXs", out)
    out = re.sub(r"(?m)^(rootdir|cachedir|plugins|platform):.*$", r"\1: <scrubbed>", out)
    return out
