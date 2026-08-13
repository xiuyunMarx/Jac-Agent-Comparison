"""Port of byLLM/tests/edit_tests.jac.

Every test writes into a throwaway root, never the real repository.
"""

from __future__ import annotations

import os

import pytest

from conftest import read, write
from tools.edit import EditCode, syntax_error

try:  # jaclang is byLLM's runtime, not a dependency here.
    import jaclang.jac0core.parser  # noqa: F401

    HAVE_JAC = True
except ImportError:  # pragma: no cover -- depends on the environment
    HAVE_JAC = False


def test_write_file_creates_a_new_file(tmp_repo: str) -> None:
    ed = EditCode(repo_root=tmp_repo)
    out = ed.write_file("a.txt", "x\ny\n")
    assert out.startswith("OK: Created a.txt")
    assert "2 lines" in out
    assert read(tmp_repo, "a.txt") == "x\ny\n"


def test_write_file_names_the_previous_line_count_when_overwriting(tmp_repo: str) -> None:
    # Overwriting must name the previous line count, so a clobber is visible.
    write(tmp_repo, "b.txt", "1\n2\n3\n4\n5\n")
    ed = EditCode(repo_root=tmp_repo)
    out = ed.write_file("b.txt", "just one line")
    assert out.startswith("OK: Overwrote b.txt")
    assert "was 5 lines" in out
    assert "now 1 lines" in out


def test_write_file_adds_a_trailing_newline(tmp_repo: str) -> None:
    # A trailing newline is added, so diffs do not churn.
    ed = EditCode(repo_root=tmp_repo)
    ed.write_file("c.txt", "no trailing newline")
    assert read(tmp_repo, "c.txt").endswith("\n")


def test_write_file_creates_missing_parent_directories(tmp_repo: str) -> None:
    # Missing parent directories are created, since there is no mkdir tool.
    ed = EditCode(repo_root=tmp_repo)
    out = ed.write_file("deep/nested/d.txt", "hi\n")
    assert out.startswith("OK: Created")
    assert "created directory" in out
    assert os.path.isfile(os.path.join(tmp_repo, "deep/nested/d.txt"))


def test_write_file_refuses_to_escape_the_root(tmp_repo: str) -> None:
    ed = EditCode(repo_root=tmp_repo)
    out = ed.write_file("../escape.txt", "should not exist")
    assert out.startswith("BLOCKED: ")
    assert not os.path.exists(os.path.join(os.path.dirname(tmp_repo), "escape.txt"))


def test_write_file_refuses_protected_directories(tmp_repo: str) -> None:
    ed = EditCode(repo_root=tmp_repo)
    assert ed.write_file(".git/config", "[diff]\n").startswith("BLOCKED: ")
    assert ed.write_file(".jac/x", "y").startswith("BLOCKED: ")
    assert not os.path.exists(os.path.join(tmp_repo, ".git/config"))


def test_write_file_refuses_to_overwrite_a_binary_file(tmp_repo: str) -> None:
    # Text must not be written over a binary file.
    with open(os.path.join(tmp_repo, "img.png"), "wb") as f:
        f.write(b"\x89PNG\x00\x00binary")
    ed = EditCode(repo_root=tmp_repo)
    assert ed.write_file("img.png", "text").startswith("Error: ")


def test_replace_in_file_replaces_a_unique_anchor(tmp_repo: str) -> None:
    write(tmp_repo, "e.txt", "alpha\nbeta\ngamma\n")
    ed = EditCode(repo_root=tmp_repo)
    out = ed.replace_in_file("e.txt", "beta", "BETA")
    assert out.startswith("OK: replaced 1 occurrence(s) in e.txt at line 2.")
    assert read(tmp_repo, "e.txt") == "alpha\nBETA\ngamma\n"


def test_replace_in_file_reports_a_missing_anchor(tmp_repo: str) -> None:
    write(tmp_repo, "f.txt", "alpha\n")
    ed = EditCode(repo_root=tmp_repo)
    out = ed.replace_in_file("f.txt", "nowhere", "x")
    assert out.startswith("Error: no match for 'old'")
    assert "read_file" in out
    assert read(tmp_repo, "f.txt") == "alpha\n"


def test_replace_in_file_hints_at_surrounding_whitespace(tmp_repo: str) -> None:
    # Near-match by surrounding whitespace: old.strip() does occur.
    write(tmp_repo, "g.txt", "alpha\nbeta\ngamma\n")
    ed = EditCode(repo_root=tmp_repo)
    out = ed.replace_in_file("g.txt", "   beta   ", "BETA")
    assert out.startswith("Error: no match for 'old'")
    assert "near-match" in out


def test_replace_in_file_hints_at_internal_spacing_drift(tmp_repo: str) -> None:
    # Near-match by internal spacing: the file uses different spacing than the
    # model remembers. This is the dominant anchored-edit failure mode.
    write(tmp_repo, "g2.txt", "def foo():\n    x  =  1\n")
    ed = EditCode(repo_root=tmp_repo)
    out = ed.replace_in_file("g2.txt", "x = 1", "x = 2")
    assert out.startswith("Error: no match for 'old'")
    assert "whitespace normalization" in out
    assert "line 2" in out


def test_replace_in_file_refuses_a_duplicated_anchor_and_leaves_the_file_intact(
    tmp_repo: str,
) -> None:
    # A duplicated anchor must refuse AND leave the file byte-identical.
    body = "x = 1\ny = 2\nx = 1\nz = 3\nx = 1\n"
    write(tmp_repo, "h.txt", body)
    ed = EditCode(repo_root=tmp_repo)
    out = ed.replace_in_file("h.txt", "x = 1", "x = 9")
    assert out.startswith("Error: found 3 occurrences")
    assert "expected_count=3" in out
    assert read(tmp_repo, "h.txt") == body


def test_replace_in_file_replaces_all_when_expected_count_matches(tmp_repo: str) -> None:
    # ... and expected_count is the sanctioned escape hatch.
    write(tmp_repo, "i.txt", "x = 1\ny = 2\nx = 1\n")
    ed = EditCode(repo_root=tmp_repo)
    out = ed.replace_in_file("i.txt", "x = 1", "x = 9", 2)
    assert out.startswith("OK: replaced 2 occurrence(s)")
    assert read(tmp_repo, "i.txt") == "x = 9\ny = 2\nx = 9\n"


def test_replace_in_file_refuses_an_empty_anchor(tmp_repo: str) -> None:
    write(tmp_repo, "j.txt", "hello\n")
    ed = EditCode(repo_root=tmp_repo)
    # Empty `old` would otherwise splice at byte 0.
    out = ed.replace_in_file("j.txt", "", "PREFIX")
    assert out.startswith("Error: 'old' must not be empty")
    assert read(tmp_repo, "j.txt") == "hello\n"


def test_replace_in_file_refuses_an_empty_anchor_on_an_empty_file(tmp_repo: str) -> None:
    write(tmp_repo, "k.txt", "")
    ed = EditCode(repo_root=tmp_repo)
    # "".count("") == 1, so an empty file is the case that would sneak past a
    # bare uniqueness check.
    out = ed.replace_in_file("k.txt", "", "PREFIX")
    assert out.startswith("Error: 'old' must not be empty")
    assert read(tmp_repo, "k.txt") == ""


def test_replace_in_file_refuses_an_identical_replacement(tmp_repo: str) -> None:
    write(tmp_repo, "l.txt", "same\n")
    ed = EditCode(repo_root=tmp_repo)
    out = ed.replace_in_file("l.txt", "same", "same")
    assert out.startswith("Error: 'old' and 'new' are identical")
    assert read(tmp_repo, "l.txt") == "same\n"


def test_replace_in_file_preserves_crlf_line_endings(tmp_repo: str) -> None:
    # CRLF normalizes in and is restored out, so the model's LF anchor matches
    # and the file's line endings survive.
    write(tmp_repo, "m.txt", "alpha\r\nbeta\r\ngamma\r\n")
    ed = EditCode(repo_root=tmp_repo)
    out = ed.replace_in_file("m.txt", "beta", "BETA")
    assert out.startswith("OK: replaced 1")
    assert read(tmp_repo, "m.txt") == "alpha\r\nBETA\r\ngamma\r\n"


def test_replace_in_file_points_at_write_file_for_a_missing_file(tmp_repo: str) -> None:
    ed = EditCode(repo_root=tmp_repo)
    out = ed.replace_in_file("missing.txt", "a", "b")
    assert out.startswith("Error: ")
    assert "write_file" in out


def test_replace_in_file_refuses_to_escape_the_root(tmp_repo: str) -> None:
    write(tmp_repo, "n.txt", "x\n")
    ed = EditCode(repo_root=tmp_repo)
    assert ed.replace_in_file("../n.txt", "x", "y").startswith("BLOCKED: ")


# ---------------------------------------------------------------------------
# Post-edit syntax check. A broken write that says "OK" sends the model looking
# for the damage through an import error three calls later; in one measured
# astropy run that search was twenty-four consecutive read_file calls walking
# backwards through a 4,473-line file one line at a time.
# ---------------------------------------------------------------------------


def test_syntax_error_finds_the_line_in_python_and_says_nothing_about_clean_code() -> None:
    assert syntax_error("def f():\n    return 1\n", "a.py") == ""
    broken = syntax_error("def f():\nreturn 1\n", "a.py")
    assert broken.startswith("line 2: IndentationError")
    assert syntax_error("def f(:\n", "a.py").startswith("line 1: SyntaxError")


@pytest.mark.skipif(not HAVE_JAC, reason="jaclang is not installed")
def test_syntax_error_finds_the_line_in_jac() -> None:
    # Jac is not a dependency of this project, but when the runtime is present
    # the two sides must judge a .jac write identically -- the action space is
    # what the comparison holds fixed.
    assert syntax_error("with entry {\n    print(1);\n}\n", "a.jac") == ""
    broken = syntax_error("with entry {\n    print(1)\n}\n", "a.jac")
    assert broken.startswith("line 3")
    assert "';'" in broken


def test_syntax_error_keeps_quiet_about_files_it_cannot_parse() -> None:
    # Not parsing a format is not the same as calling it broken.
    assert syntax_error("this is not code at all {{{", "notes.md") == ""
    assert syntax_error("}{", "data.json") == ""


def test_write_file_warns_when_the_write_breaks_the_parse(tmp_repo: str) -> None:
    ed = EditCode(repo_root=tmp_repo)
    out = ed.write_file("m.py", "def f():\nreturn 1\n")
    # The write is still a write -- reverting behind the model's back would
    # leave it reasoning about a file that is not the one on disk.
    assert out.startswith("OK: Created m.py")
    assert "WARNING" in out
    assert "line 2" in out
    assert read(tmp_repo, "m.py") == "def f():\nreturn 1\n"
    # Clean code says nothing.
    assert "WARNING" not in ed.write_file("n.py", "def f():\n    return 1\n")


def test_replace_in_file_warns_when_the_edit_breaks_the_parse(tmp_repo: str) -> None:
    # The exact shape of the astropy run's failure: an anchored edit that
    # dedents a method body, reported as OK and found three calls later.
    write(tmp_repo, "m.py", "class A:\n    def f(self):\n        return 1\n")
    ed = EditCode(repo_root=tmp_repo)
    out = ed.replace_in_file("m.py", "        return 1", "    return 1")
    assert out.startswith("OK: replaced 1 occurrence(s)")
    assert "WARNING" in out
    assert "m.py no longer parses" in out
    # A sound edit to the same file says nothing.
    write(tmp_repo, "k.py", "class A:\n    def f(self):\n        return 1\n")
    assert "WARNING" not in ed.replace_in_file("k.py", "return 1", "return 2")
