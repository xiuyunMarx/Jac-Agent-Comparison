"""Port of byLLM/tests/plan_tests.jac.

byLLM passes a `TaskStatus` enum member; here the status is the `Literal` string
the JSON schema constrains it to, which is what reaches the method on both sides.
"""

from __future__ import annotations

from tools.plan import PlanTasks


def test_an_empty_board_renders_a_sentinel() -> None:
    board = PlanTasks()
    assert "no plan yet" in board.show_plan()
    assert board.show_plan() != ""


def test_set_plan_renders_a_numbered_checklist() -> None:
    board = PlanTasks()
    out = board.set_plan(["read the file", "fix the bug", "run the tests"])
    assert out.startswith("OK: plan set with 3 steps.")
    assert "[ ] 1. read the file" in out
    assert "[ ] 3. run the tests" in out
    assert "(0/3 done)" in out


def test_update_task_marks_a_step_done() -> None:
    board = PlanTasks()
    board.set_plan(["one", "two"])
    out = board.update_task(1, "done")
    assert out.startswith("OK: step 1 is now done.")
    assert "[x] 1. one" in out
    assert "(1/2 done)" in out


def test_update_task_records_doing_and_blocked_with_a_note() -> None:
    board = PlanTasks()
    board.set_plan(["one", "two"])
    doing = board.update_task(2, "doing")
    assert "[~] 2. two" in doing
    blocked = board.update_task(2, "blocked", "waiting on the schema")
    assert "[!] 2. two" in blocked
    assert "waiting on the schema" in blocked


def test_update_task_reports_an_unknown_id_with_the_known_ids() -> None:
    board = PlanTasks()
    board.set_plan(["one", "two"])
    out = board.update_task(99, "done")
    assert out.startswith("Error: no step with id 99.")
    assert "Known ids: 1, 2." in out


def test_update_task_points_at_set_plan_when_no_plan_exists() -> None:
    board = PlanTasks()
    out = board.update_task(1, "done")
    assert out.startswith("Error: ")
    assert "set_plan" in out


def test_set_plan_rejects_an_empty_step_list() -> None:
    board = PlanTasks()
    assert board.set_plan([]).startswith("Error: ")
    assert board.set_plan(["   ", ""]).startswith("Error: ")


def test_set_plan_replaces_the_plan_and_restarts_numbering() -> None:
    board = PlanTasks()
    board.set_plan(["a", "b", "c"])
    board.update_task(1, "done")
    out = board.set_plan(["x"])
    assert "(0/1 done)" in out
    assert "[ ] 1. x" in out
    assert "a" not in out.split("\n")[1]


def test_show_plan_refetches_the_current_state() -> None:
    # show_plan re-fetches state after transcript compaction.
    board = PlanTasks()
    board.set_plan(["alpha"])
    board.update_task(1, "doing", "in flight")
    shown = board.show_plan()
    assert "[~] 1. alpha" in shown
    assert "in flight" in shown


def test_the_status_parameter_is_a_json_schema_enum() -> None:
    # byLLM used a `TaskStatus` enum purely to constrain the parameter; the
    # Literal has to produce the same constraint or the model sees a free string.
    board = PlanTasks()
    schema = board.update_task_tool().args
    assert schema["status"]["enum"] == ["todo", "doing", "done", "blocked"]
