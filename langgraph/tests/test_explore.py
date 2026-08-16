"""Port of byLLM/tests/explore_tests.jac.

The byLLM tests read the byLLM project's own sources; these read this project's,
so `nodes/explore.jac` becomes `tools/explore.py` and `node ExploreCodeBase`
becomes `class ExploreCodeBase`.
"""

from __future__ import annotations

import io
import os

from conftest import PROJECT_ROOT, write
from tools.common import get_tool_calls, reset_tool_log
from tools.explore import ExploreCodeBase

PROJ = PROJECT_ROOT


def big_repo(root: str, lines: int = 2000) -> int:
    """Write a file too big to serve whole into `root`; return its line count.

    Generated rather than pointed at a project file: which files sit either side
    of the whole-file budget drifts every time one of them is edited, and a
    windowing test that silently stops testing windowing is worse than no test.
    """
    write(root, "big.txt", "\n".join(
        f"line {i:05d} " + "x" * 40 for i in range(1, lines + 1)
    ) + "\n")
    return lines


def line_count(rel: str) -> int:
    with io.open(os.path.join(PROJ, rel), encoding="utf-8") as f:
        return len(f.read().splitlines())


def test_read_file_returns_content_with_a_range_header() -> None:
    ex = ExploreCodeBase(repo_root=PROJ)
    out = ex.read_file("tools/explore.py")
    assert isinstance(out, str)
    assert out.startswith("# tools/explore.py - lines 1-")
    assert "class ExploreCodeBase" in out


def test_read_file_reports_a_missing_file_as_actionable_text() -> None:
    # The regression test for the print-and-return-"" failure mode: a failure
    # must reach the model as non-empty, actionable text.
    ex = ExploreCodeBase(repo_root=PROJ)
    out = ex.read_file("tools/does_not_exist.py")
    assert out.startswith("Error: ")
    assert len(out) > 20
    assert "find_files" in out


def test_read_tools_refuse_paths_outside_the_root() -> None:
    ex = ExploreCodeBase(repo_root=PROJ)
    assert ex.read_file("../../etc/passwd").startswith("BLOCKED: ")
    assert ex.read_file("/etc/passwd").startswith("BLOCKED: ")
    assert ex.ls_repo("..").startswith("BLOCKED: ")


def test_read_file_rejects_a_directory_target() -> None:
    ex = ExploreCodeBase(repo_root=PROJ)
    assert ex.read_file("tools").startswith("Error: ")
    assert "ls_repo" in ex.read_file("tools")


def test_read_file_describes_an_empty_file_instead_of_returning_nothing(
    tmp_repo: str,
) -> None:
    # An empty file must not return "" -- the model cannot act on that.
    write(tmp_repo, "empty.txt", "")
    ex = ExploreCodeBase(repo_root=tmp_repo)
    out = ex.read_file("empty.txt")
    assert out != ""
    assert "empty" in out


def test_read_file_honours_a_line_window_on_a_file_too_big_to_serve_whole(
    tmp_repo: str,
) -> None:
    total = big_repo(tmp_repo)
    ex = ExploreCodeBase(repo_root=tmp_repo)
    windowed = ex.read_file("big.txt", 2, 4)
    assert windowed.startswith(f"# big.txt - lines 2-4 of {total}")
    assert len(windowed.splitlines()) == 4
    past = ex.read_file("big.txt", 100000)
    assert past.startswith("Error: ")


def test_a_file_that_fits_comes_back_whole_whatever_window_was_asked_for() -> None:
    # Windowing a small file buys nothing and costs the loop guard its teeth:
    # the served-window memory keys on the range, so a dozen overlapping
    # windows of one small file are a dozen fresh reads and none of them a
    # repeat it can refuse. That is the shape the read_file scan loops took.
    total = line_count("tools/plan.py")
    ex = ExploreCodeBase(repo_root=PROJ)
    assert ex.read_file("tools/plan.py").startswith(
        f"# tools/plan.py - lines 1-{total} of {total}"
    )
    ex.set_scope("second")
    assert ex.read_file("tools/plan.py", 40, 60).startswith(
        f"# tools/plan.py - lines 1-{total} of {total}"
    )
    # A window past the end is still an error, not quietly forgiven.
    ex.set_scope("third")
    assert ex.read_file("tools/plan.py", 100000).startswith("Error: ")


def test_ls_repo_lists_sorted_entries() -> None:
    ex = ExploreCodeBase(repo_root=PROJ)
    listing = ex.ls_repo("tools")
    assert "explore.py" in listing
    assert "common.py" in listing
    # sorted order is load-bearing for benchmark determinism
    names = [n for n in listing.splitlines() if n.endswith(".py")]
    assert names == sorted(names)


def test_ls_repo_hides_skipped_directories(tmp_repo: str) -> None:
    write(tmp_repo, "src/main.py", "x = 1\n")
    write(tmp_repo, "__pycache__/main.pyc", "junk\n")
    os.makedirs(os.path.join(tmp_repo, ".git"), exist_ok=True)
    ex = ExploreCodeBase(repo_root=tmp_repo)
    listing = ex.ls_repo(".")
    assert "src/" in listing
    assert "__pycache__/" not in listing
    assert ".git/" not in listing


def test_ls_repo_names_an_empty_directory(tmp_repo: str) -> None:
    os.makedirs(os.path.join(tmp_repo, "sub"))
    ex = ExploreCodeBase(repo_root=tmp_repo)
    assert "empty directory" in ex.ls_repo("sub")


def test_find_files_locates_by_name_and_by_path_glob() -> None:
    ex = ExploreCodeBase(repo_root=PROJ)
    found = ex.find_files("*.py", ".")
    assert "tools/verify.py" in found
    assert "tools/plan.py" in found
    scoped = ex.find_files("tools/*.py", ".")
    assert "tools/edit.py" in scoped
    assert ex.find_files("*.nosuchext", ".").startswith("No files matching")


def test_grep_finds_a_symbol_across_the_tree() -> None:
    ex = ExploreCodeBase(repo_root=PROJ)
    hits = ex.grep("class ExploreCodeBase", "tools", "*.py")
    assert "explore.py:" in hits
    assert isinstance(hits, str)


def test_grep_with_no_matches_returns_a_sentence(tmp_repo: str) -> None:
    # No matches must be a sentence, never "[]" or "". Searched in a temp dir so
    # the pattern literal in this test file cannot match itself.
    write(tmp_repo, "a.txt", "nothing interesting here\n")
    ex = ExploreCodeBase(repo_root=tmp_repo)
    out = ex.grep("absent_symbol", ".")
    assert out.startswith("No matches for")
    assert "searched 1 files" in out


def test_grep_reports_an_invalid_regex_with_a_recovery_hint() -> None:
    ex = ExploreCodeBase(repo_root=PROJ)
    bad = ex.grep("def (", "tools")
    assert bad.startswith("Error: invalid regular expression")
    assert "(?i)" in bad


def test_grep_supports_inline_case_insensitive_flag() -> None:
    # ignore_case was dropped in favour of the inline flag; prove it works.
    ex = ExploreCodeBase(repo_root=PROJ)
    hits = ex.grep("(?i)CLASS EXPLORECODEBASE", "tools", "*.py")
    assert "explore.py:" in hits


def test_grep_emits_repository_relative_paths() -> None:
    # Every returned path is repo-relative, so output does not depend on where
    # the benchmark is checked out.
    ex = ExploreCodeBase(repo_root=PROJ)
    hits = ex.grep("import os", "tools", "*.py")
    for line in hits.splitlines():
        assert not line.startswith("/")


# --- served-window memory ----------------------------------------------------
#
# These cover the loop that cost a whole benchmark run: the model read one file,
# could not tell the result was the same one it already had, and read it again,
# 13 times over, until the phase budget ran out.


def test_read_file_answers_a_repeat_with_the_next_move_not_the_same_bytes(
    tmp_repo: str,
) -> None:
    total = big_repo(tmp_repo)
    ex = ExploreCodeBase(repo_root=tmp_repo, scope="Exploring")
    first = ex.read_file("big.txt", 1, 20)
    assert first.startswith(f"# big.txt - lines 1-20 of {total}")
    again = ex.read_file("big.txt", 1, 20)
    assert again.startswith("Error: ")
    assert "already read" in again
    # A refusal that names no next move is just a different dead end.
    assert "read_file(file_path='big.txt', start_line=21)" in again
    assert "grep" in again


def test_a_repeat_of_a_fully_read_file_is_sent_somewhere_other_than_the_file() -> None:
    total = line_count("tools/plan.py")
    ex = ExploreCodeBase(repo_root=PROJ, scope="Exploring")
    whole = ex.read_file("tools/plan.py")
    assert whole.startswith(f"# tools/plan.py - lines 1-{total} of {total}")
    again = ex.read_file("tools/plan.py")
    assert again.startswith("Error: ")
    assert f"all {total} lines" in again
    assert "grep" in again


def test_each_phase_gets_its_own_served_window_memory() -> None:
    # Phases keep separate conversations, so a window Exploring read is
    # genuinely absent from Editing's messages and Editing must still be served
    # it.
    total = line_count("tools/plan.py")
    ex = ExploreCodeBase(repo_root=PROJ, scope="Exploring")
    assert ex.read_file("tools/plan.py").startswith("# tools/plan.py")
    ex.set_scope("Editing")
    out = ex.read_file("tools/plan.py")
    assert out.startswith(f"# tools/plan.py - lines 1-{total} of {total}")


def test_writing_the_file_re_opens_it_for_reading(tmp_repo: str) -> None:
    # Editing reads, edits, then reads back to check the edit landed. The
    # size+mtime stamp in the key expires the entry on its own, so no tool has
    # to remember to invalidate anything.
    write(tmp_repo, "a.txt", "one\ntwo\n")
    ex = ExploreCodeBase(repo_root=tmp_repo, scope="Editing")
    assert "one" in ex.read_file("a.txt")
    assert ex.read_file("a.txt").startswith("Error: ")
    write(tmp_repo, "a.txt", "one\ntwo\nthree\n")
    after = ex.read_file("a.txt")
    assert after.startswith("# a.txt - lines 1-3 of 3")
    assert "three" in after


def test_an_oversized_file_pages_forward_on_whole_lines(tmp_repo: str) -> None:
    # The clip-at-a-character-offset form reported "N more characters", which
    # names no line -- so no start_line resumed it and repeating the identical
    # call was the only move left.
    line = "x" * 200
    write(tmp_repo, "big.txt", "\n".join(line for _ in range(400)) + "\n")
    ex = ExploreCodeBase(repo_root=tmp_repo, scope="Exploring")
    page1 = ex.read_file("big.txt")
    assert "Continue with read_file(" in page1
    # Every content line arrived whole: replace_in_file anchors on exact text.
    for ln in page1.splitlines()[1:]:
        assert ln == line or ln.startswith("... [stopped")
    resume = int(page1.split("start_line=")[-1].strip().rstrip(")."))
    assert resume > 1
    page2 = ex.read_file("big.txt", resume)
    assert page2.startswith(f"# big.txt - lines {resume}-")


def test_a_single_over_budget_line_still_advances(tmp_repo: str) -> None:
    # One line longer than the whole budget must not return an empty window
    # whose continuation hint points back at the call that produced it.
    write(tmp_repo, "wide.txt", ("y" * 60000) + "\nshort\n")
    ex = ExploreCodeBase(repo_root=tmp_repo, scope="Exploring")
    page1 = ex.read_file("wide.txt")
    assert page1.startswith("# wide.txt - lines 1-1 of 2")
    assert "start_line=2" in page1
    assert ex.read_file("wide.txt", 2).startswith("# wide.txt - lines 2-2 of 2")


def test_read_file_logs_the_window_it_served_not_just_the_path() -> None:
    # Without start_line/end_line in the log, a run of paging calls and a run of
    # identical calls are the same sequence of entries -- which is how the loop
    # stayed invisible in the run log.
    reset_tool_log()
    ex = ExploreCodeBase(repo_root=PROJ, scope="Exploring")
    ex.read_file("tools/plan.py", 5, 9)
    calls = get_tool_calls()
    assert len(calls) == 1
    assert calls[0].args["file_path"] == "tools/plan.py"
    assert calls[0].args["start_line"] == "5"
    assert calls[0].args["end_line"] == "9"
    reset_tool_log()


def test_read_file_names_the_fix_for_a_backwards_window() -> None:
    ex = ExploreCodeBase(repo_root=PROJ, scope="Exploring")
    out = ex.read_file("tools/plan.py", 20, 5)
    assert out.startswith("Error: ")
    assert "end_line" in out
    assert "at least 20" in out
