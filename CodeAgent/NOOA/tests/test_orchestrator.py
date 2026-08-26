"""The run: the report, the repair loop, the budget, and what the shim reads.

Every test here drives the real agent -- the real session, the real REPL, the
real tools -- on a model whose turns are scripted, so nothing leaves the
process.
"""

from __future__ import annotations

from CodeAgent.NOOA.tests.conftest import FAKE_USAGE, report_cell, scripted

import CodeAgent.NOOA.llm as L
import CodeAgent.NOOA.orchestrator as O

# The shortest honest run: look, edit, run something, report. Each string is one
# cell the scripted model answers with.
GOOD = (
    "print(self.grep('hello'))",
    "print(self.replace_in_file('pkg/greet.py', '\"hello\"', '\"world\"'))",
    "print(self.run_command('git diff'))",
    report_cell(verify_command="git diff", evidence="the diff shows world"),
)


def run(repo, *cells, max_steps=10, objective="make greet return world"):
    L.set_client(scripted(*cells))
    return O.solve(objective, str(repo), max_steps)


# --- the happy path --------------------------------------------------------


def test_a_run_edits_the_workspace(repo):
    run(repo, *GOOD)
    assert (repo / "pkg" / "greet.py").read_text() == 'def greet():\n    return "world"\n'


def test_the_answer_is_the_rendered_report(repo):
    result = run(repo, *GOOD)
    assert "resolved: True" in result.answer
    assert "## Files changed" in result.answer
    assert "## Evidence" in result.answer
    assert result.resolved


def test_one_attempt_is_one_step(repo):
    assert run(repo, *GOOD).steps == 1


def test_the_tool_log_is_ordered_and_includes_the_gates_re_run(repo):
    result = run(repo, *GOOD)
    assert [c.name for c in result.tool_calls] == [
        "grep", "replace_in_file", "run_command", "run_command"
    ]


def test_the_totals_are_the_sum_of_the_calls(repo):
    result = run(repo, *GOOD)
    assert result.llm_calls == len(GOOD)
    assert result.prompt_tokens == result.llm_calls * FAKE_USAGE["prompt_tokens"]
    assert result.completion_tokens == result.llm_calls * FAKE_USAGE["completion_tokens"]
    # A subset of prompt_tokens, never an addition to it.
    assert result.cached_tokens == result.llm_calls * 40 < result.prompt_tokens


def test_the_shim_reads_the_field_set_it_expects(repo):
    result = run(repo, *GOOD)
    for field in ("answer", "steps", "llm_calls", "prompt_tokens",
                  "completion_tokens", "cached_tokens", "tool_calls"):
        assert hasattr(result, field), field


# --- the gate, end to end --------------------------------------------------


def test_a_report_whose_command_fails_is_repaired_then_accepted(repo):
    """First attempt names a command that exits non-zero; the repair names one
    that passes. Two steps, and the answer is the second report."""
    result = run(
        repo,
        # attempt one
        "print(self.write_file('pkg/new.py', 'x = 1\\n'))",
        "print(self.run_command('git status'))",
        report_cell(verify_command="git ls-files --error-unmatch pkg/new.py",
                    evidence="wrote it"),
        # repair
        "print(self.run_command('git status'))",
        report_cell(verify_command="git status", evidence="git status is clean-ish"),
    )
    assert result.steps == 2
    assert result.resolved
    assert "git status is clean-ish" in result.answer


def test_a_claim_the_gate_refuses_is_downgraded_not_believed(repo):
    """The budget runs out with the gate still failing, so the answer says so
    even though the model's last report claimed otherwise."""
    result = run(
        repo,
        "print(self.write_file('pkg/new.py', 'x = 1\\n'))",
        "print(self.run_command('git status'))",
        report_cell(verify_command="git ls-files --error-unmatch nope.py"),
        max_steps=1,
    )
    assert not result.resolved
    assert "resolved: False" in result.answer
    assert "re-checked" in result.answer


def test_an_honest_failure_is_not_re_run(repo):
    """A report that admits it failed is repaired on its own account, so the
    gate does not spend a test run proving what it already said."""
    result = run(
        repo,
        report_cell(resolved=False, files_changed=[], verify_command="git status",
                    evidence="could not find the symbol"),
        max_steps=1,
    )
    assert not result.resolved
    assert [c.name for c in result.tool_calls] == []


# --- the budget ------------------------------------------------------------


def test_the_budget_bounds_the_repair_loop(repo):
    """Three attempts, each failing its gate, with a budget of two."""
    attempt = (
        "print(self.write_file('pkg/new.py', 'x = 1\\n'))",
        "print(self.run_command('git status'))",
        report_cell(verify_command="git ls-files --error-unmatch nope.py"),
    )
    result = run(repo, *(attempt * 3), max_steps=2)
    assert result.steps == 2
    assert not result.resolved


# --- failure modes ---------------------------------------------------------


def test_a_session_that_dies_still_answers(repo):
    """The scripted model runs out of turns mid-session, so the generation
    raises. The run must still report, because the driver takes a patch from the
    workspace either way."""
    result = run(repo, "print(self.grep('hello'))")
    assert "the run failed" in result.answer
    assert "## Plan" in result.answer


# --- the seam --------------------------------------------------------------


def test_the_model_name_answers_without_a_client(monkeypatch):
    monkeypatch.delenv("CODEAGENT_MODEL", raising=False)
    L.set_client(None)
    L.set_model(None)
    assert L.active_model_name() == L.DEFAULT_MODEL
    monkeypatch.setenv("CODEAGENT_MODEL", "ollama_chat/qwen3")
    L.set_model(None)
    # The provider prefix survives: this side is litellm-routed.
    assert L.active_model_name() == "ollama_chat/qwen3"


def test_metering_a_client_twice_does_not_double_count(repo):
    L.set_client(scripted(report_cell(resolved=False, files_changed=[],
                                      verify_command="", evidence="nothing to do")))
    L.get_client()
    L.get_client()
    assert O.solve("nothing", str(repo), 1).llm_calls == 1
