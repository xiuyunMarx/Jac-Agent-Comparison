"""Port of byLLM/tests/common_tests.jac."""

from __future__ import annotations

import os
import tempfile

from tools.common import (
    clip,
    clip_ends,
    get_tool_calls,
    is_write_blocked,
    log_tool_call,
    reset_tool_log,
    safe_path,
    scrub,
)


def test_safe_path_accepts_a_path_inside_the_root(tmp_repo: str) -> None:
    inside = safe_path(tmp_repo, "a/b.txt")
    assert inside is not None
    assert str(inside).startswith(tmp_repo)


def test_safe_path_rejects_traversal_and_absolute_escapes(tmp_repo: str) -> None:
    assert safe_path(tmp_repo, "../../etc/passwd") is None
    assert safe_path(tmp_repo, "/etc/passwd") is None
    assert safe_path(tmp_repo, "a/../../escape") is None


def test_safe_path_rejects_a_symlink_pointing_outside_the_root(tmp_repo: str) -> None:
    # A symlink created inside the root pointing outside it must be refused.
    # This is what realpath buys over abspath.
    outside = os.path.realpath(tempfile.mkdtemp())
    os.symlink(outside, os.path.join(tmp_repo, "link"))
    assert safe_path(tmp_repo, "link/secret.txt") is None


def test_safe_path_rejects_a_sibling_directory_sharing_the_prefix(tmp_repo: str) -> None:
    # commonpath compares segments; startswith would accept the sibling.
    root_dir = os.path.join(tmp_repo, "repo")
    evil = os.path.join(tmp_repo, "repo-evil")
    os.makedirs(root_dir)
    os.makedirs(evil)
    assert safe_path(root_dir, os.path.join(evil, "x.txt")) is None


def test_is_write_blocked_guards_dot_git_and_dot_jac(tmp_repo: str) -> None:
    git_cfg = safe_path(tmp_repo, ".git/config")
    assert git_cfg is not None
    assert is_write_blocked(tmp_repo, str(git_cfg))
    ok_file = safe_path(tmp_repo, "tools/x.py")
    assert not is_write_blocked(tmp_repo, str(ok_file))


def test_clip_and_clip_ends_mark_what_they_dropped() -> None:
    short = "hello"
    assert clip(short, 100) == short
    assert clip_ends(short, 100) == short
    long_text = "x" * 500
    clipped = clip(long_text, 100)
    assert len(clipped) < 500
    assert "truncated" in clipped
    both = clip_ends(long_text, 100)
    assert "omitted" in both
    assert both.startswith("x")
    assert both.endswith("x")


def test_scrub_removes_absolute_paths_and_durations(tmp_repo: str) -> None:
    text = f"ran in {tmp_repo}/tools took 1.23s"
    out = scrub(text, tmp_repo)
    assert tmp_repo not in out
    assert "X.XXs" in out


def test_tool_call_log_records_and_resets() -> None:
    reset_tool_log()
    log_tool_call("read_file", {"file_path": "a.py"})
    log_tool_call("grep", {"pattern": "x"})
    calls = get_tool_calls()
    assert len(calls) == 2
    assert calls[0].name == "read_file"
    assert calls[1].args["pattern"] == "x"
    reset_tool_log()
    assert len(get_tool_calls()) == 0
