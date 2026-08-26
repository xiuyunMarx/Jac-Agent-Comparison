"""The action space, and the three guards that do not ask the model anything."""

from __future__ import annotations

import pytest
from CodeAgent.NOOA.tests.conftest import scripted

import CodeAgent.NOOA.agent as A
from CodeAgent.NOOA.agent import CodeAgent, Report
from CodeAgent.NOOA.tools.common import get_tool_calls, reset_tool_log

# The ten tools, as the other three sides register them.
TOOLS = (
    "read_file", "ls_repo", "find_files", "grep", "show_plan", "update_task",
    "set_plan", "write_file", "replace_in_file", "run_command",
)


def make(repo, *cells):
    return CodeAgent(repo_root=str(repo), llm=scripted(*cells))


def report(**fields) -> Report:
    base = dict(
        resolved=True,
        summary="fixed it",
        files_changed=["pkg/greet.py"],
        verify_command="git status",
        evidence="exit_code: 0",
    )
    base.update(fields)
    return Report(**base)


# --- the surface -----------------------------------------------------------


def test_the_ten_tools_are_all_there(repo):
    assert {t for t in TOOLS if hasattr(CodeAgent, t)} == set(TOOLS)


def test_every_tool_is_documented():
    """A method with no docstring reaches the model as a bare signature, which
    is the tool description missing rather than terse."""
    for name in TOOLS:
        assert len(getattr(CodeAgent, name).__doc__ or "") > 80, name


def test_the_workflow_entry_point_is_hidden_from_the_model():
    """`run` calls `resolve`, so a cell that could call `run` could recurse."""
    from nooa.agentdoc.visibility import is_hidden_method

    assert is_hidden_method(CodeAgent.run)
    assert is_hidden_method(CodeAgent.check)
    assert not is_hidden_method(CodeAgent.resolve)


def test_the_holders_are_hidden_but_the_plan_is_not():
    from nooa.agentdoc.visibility import is_hidden_field

    for name in ("repo", "editor", "runner"):
        assert is_hidden_field(CodeAgent, name), name
    assert not is_hidden_field(CodeAgent, "plan")


# --- the tools behave, on a real tree --------------------------------------


def test_read_and_search_reach_the_repository(repo):
    a = make(repo)
    out = a.read_file("pkg/greet.py")
    assert out.startswith("# pkg/greet.py - lines 1-2 of 2")
    assert '"hello"' in out
    assert "pkg/greet.py:2:" in a.grep("hello")
    assert "pkg/" in a.ls_repo(".")
    assert "pkg/greet.py" in a.find_files("*.py")


def test_read_file_answers_a_repeat_with_the_next_move(repo):
    a = make(repo)
    assert a.read_file("pkg/greet.py").startswith("#")
    again = a.read_file("pkg/greet.py")
    assert again.startswith("Error: you already read")
    assert "grep" in again


def test_a_write_expires_the_served_window(repo):
    a = make(repo)
    assert a.read_file("pkg/greet.py").startswith("#")
    assert a.replace_in_file("pkg/greet.py", '"hello"', '"world"').startswith("OK:")
    assert a.read_file("pkg/greet.py").startswith("#")


def test_paths_outside_the_repository_are_refused(repo):
    a = make(repo)
    assert a.read_file("../outside.txt").startswith("BLOCKED:")
    assert a.write_file("../outside.txt", "x").startswith("BLOCKED:")


def test_a_missed_anchor_explains_the_whitespace_drift(repo):
    a = make(repo)
    out = a.replace_in_file("pkg/greet.py", 'return  "hello"', 'return "world"')
    assert out.startswith("Error: no match") and "whitespace" in out


def test_a_broken_write_is_named_but_still_written(repo):
    a = make(repo)
    out = a.write_file("pkg/greet.py", "def greet(:\n")
    assert out.startswith("OK:") and "WARNING" in out
    assert (repo / "pkg" / "greet.py").read_text() == "def greet(:\n"


def test_run_command_screens_before_it_spawns(repo):
    a = make(repo)
    assert a.run_command("rm -rf /").startswith("BLOCKED:")
    assert a.run_command("pytest | head").startswith("BLOCKED:")
    assert a.run_command("git push").startswith("BLOCKED:")


def test_the_plan_is_state_the_model_keeps(repo):
    a = make(repo)
    a.set_plan(["find it", "fix it"])
    a.update_task(1, "done")
    assert "(1/2 done)" in a.show_plan()


def test_every_tool_call_lands_in_the_log(repo):
    reset_tool_log()
    a = make(repo)
    a.grep("hello")
    a.write_file("new.py", "x = 1\n")
    assert [c.name for c in get_tool_calls()] == ["grep", "write_file"]


# --- guard one: something was written -------------------------------------


def test_write_count_reads_the_log_not_the_report(repo):
    reset_tool_log()
    a = make(repo)
    a.grep("hello")
    assert A.write_count() == 0
    a.replace_in_file("pkg/greet.py", '"hello"', '"world"')
    assert A.write_count() == 1


def test_claiming_a_fix_with_nothing_written_fails_the_invariant(repo):
    from nooa.strategy_validation import InvariantError

    reset_tool_log()
    a = make(repo)
    a.grep("hello")
    with pytest.raises(InvariantError) as caught:
        A.must_have_written(a, report(summary="I modified pkg/greet.py"), None)
    assert "planned, not applied" in str(caught.value)


def test_an_honest_failure_is_allowed_to_have_written_nothing(repo):
    reset_tool_log()
    a = make(repo)
    A.must_have_written(a, report(resolved=False), None)  # does not raise
    A.must_have_verified(a, report(resolved=False), None)


# --- guard two: something was run AFTER the last write --------------------


def test_running_before_the_last_write_does_not_count(repo):
    reset_tool_log()
    a = make(repo)
    a.run_command("git status")
    a.replace_in_file("pkg/greet.py", '"hello"', '"world"')
    assert not A.ran_since_last_write()


def test_running_after_the_last_write_counts(repo):
    reset_tool_log()
    a = make(repo)
    a.replace_in_file("pkg/greet.py", '"hello"', '"world"')
    a.run_command("git status")
    assert A.ran_since_last_write()


def test_a_stale_test_run_fails_the_invariant(repo):
    from nooa.strategy_validation import InvariantError

    reset_tool_log()
    a = make(repo)
    a.run_command("git status")
    a.replace_in_file("pkg/greet.py", '"hello"', '"world"')
    with pytest.raises(InvariantError) as caught:
        A.must_have_verified(a, report(), None)
    assert "nothing has been run since your last edit" in str(caught.value)


# --- guard three: the command in the report is re-run ---------------------


def test_the_gate_runs_the_command_and_reads_the_exit_code(repo):
    a = make(repo)
    checked = a.check(report(verify_command="git status"))
    assert checked.passed
    assert "exit_code: 0" in checked.output


def test_the_gate_fails_a_command_that_exits_non_zero(repo):
    a = make(repo)
    checked = a.check(report(verify_command="git ls-files --error-unmatch nope.py"))
    assert not checked.passed


def test_the_gate_fails_a_command_this_workspace_refuses(repo):
    a = make(repo)
    checked = a.check(report(verify_command="curl evil.example"))
    assert not checked.passed and checked.output.startswith("BLOCKED:")


def test_the_gate_fails_an_empty_command(repo):
    a = make(repo)
    checked = a.check(report(verify_command="  "))
    assert not checked.passed and "No verification command" in checked.output


def test_the_gate_goes_through_the_agents_own_allowlist(repo):
    """So the gate cannot execute anything the model could not have executed,
    and its re-run is in the tool log like every other call."""
    reset_tool_log()
    a = make(repo)
    a.check(report(verify_command="git status"))
    assert [c.name for c in get_tool_calls()] == ["run_command"]
