# SPDX-License-Identifier: Apache-2.0
"""The coding agent: one object, ten capabilities, two generated methods.

The other three sides of this comparison decompose the task into a graph before
the model sees it. Five phases -- Planning, Exploring, Editing, Verifying,
Finished -- each granted a subset of the tools, wired together by edges, walked
by something that asks the model at every fork which edge to take. byLLM spells
it as a walker over `edge Flow: Phase --> Phase`; LangGraph compiles a
`StateGraph`; openai_sdk keeps the topology in a dict and walks it with a
`while` loop. All three then run a ReAct loop inside each phase, and all three
pay for a router: a second class of LLM call whose only job is to answer
"Editing or Finished?".

None of that is here.

A NOOA agent acts by writing Python in a session that has `self` in scope, so
the capabilities are methods and the workflow is code the model writes: grep for
the symbol, read the two files it points at, edit one, run the tests, read the
failure, edit again. That is one session over one object -- no phase to be in,
no edge to choose, no per-phase conversation to rebuild, and no summary to hand
forward, because the conversation that made the edit is the one still running.

So the structure here is the one NOOA's own guidance describes: methods with a
signature and a return contract for the parts that need model judgement, plain
Python for everything that must be guaranteed, and a `@hidden` entry point
chaining them. Two generated methods, `resolve` and `repair`, and three guards
that are not questions put to the model:

  * `must_have_written` -- the run cannot claim a fix having written nothing;
  * `must_have_verified` -- it cannot claim one nothing was run against *since*
    the last edit;
  * `check`, in `run` -- the command the report offers as proof is executed, by
    us, and the exit code decides. "Do not ask the model to assert that it
    verified something and treat the assertion as evidence."

The first two are NOOA postconditions, so a failure comes back as
`InvariantError` and lands in the session the model is already in: it fixes the
omission with the whole conversation still around it, rather than being routed
back to a phase whose conversation starts empty. The third is the one guard no
other side of this comparison has, because a phase in a graph reports and moves
on -- there is nothing left afterwards to re-run what it claimed.

The ten methods are the same ten tools the other sides register, with the same
prose and the same string contracts, because an identical action space is the
premise of the comparison. NOOA ships `nooa.tools.shell_tools.ShellTools` and
`nooa.tools.todo.TodoManager`, which would be the idiomatic choice for a fresh
agent and are deliberately not used here: they are not the same action space,
and a difference in the score has to be a difference between the frameworks.
What is deleted rather than ported is the schema layer -- byLLM's `sem` strings,
LangGraph's `Field(description=...)` argument models, openai_sdk's hand-written
JSON Schema -- because here the docstring and the annotations are what the model
is shown.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from nooa import Agent, CodeActStrategy, hidden, strategy
from nooa.config import CodeActConfig
from nooa.strategy_validation import InvariantError
from pydantic import BaseModel, Field

from tools.common import get_tool_calls
from tools.edit import EditCode
from tools.explore import ExploreCodeBase
from tools.plan import PlanTasks
from tools.verify import VerifyCode

# How many cells one generated method may run. byLLM's `max_react_iterations`,
# and the ceiling the other three sides put on one phase. What it counts is not
# the same thing: an iteration there is one tool call, an iteration here is one
# *cell*, and a cell can call six tools in a loop -- so the same number buys
# this side more work. It is kept at 25 anyway, because the number the others
# chose is the one worth being compared against.
CELLS_PER_STEP: int = 25

# How many generated-method calls one run may spend, when the caller does not
# say. `--max-steps` on the benchmark driver means "phase budget" on the other
# sides; here it bounds `resolve` plus its repairs, so 10 steps of 25 cells is
# the same ceiling expressed against the same numbers.
DEFAULT_MAX_STEPS: int = 10

# The only two calls that change the workspace. Every other call, however many
# the run makes, leaves the tree exactly as it was found.
WRITE_TOOLS = ("write_file", "replace_in_file")
# The only call that executes anything.
RUN_TOOLS = ("run_command",)

# `run_command` reports a clean spawn as "exit_code: N", and anything else --
# a refusal, a timeout, an output limit -- as a line that is not this one.
EXIT_CODE = re.compile(r"^exit_code: (-?\d+)$", re.MULTILINE)

# What the run is told when it reports a fix it never issued. It has to name the
# specific omission: a model told only "try again" repeats itself, because from
# inside the conversation it already believes the edit was made. byLLM's
# WRITE_GUARD_NOTE, which that side can only act on by routing its walker back
# to Editing.
NO_WRITE_NOTE = (
    "GUARD: no write_file or replace_in_file call has been made, so the "
    "workspace is still exactly as it was found. Whatever change you have "
    "described was planned, not applied. Issue the edit now, then run something "
    "against it, then report."
)

# What it is told when it reports a fix that nothing was run against *since* the
# last write. The other sides check whether a write happened at all, because
# their Verifying phase is a node in a graph rather than an event in a log, and
# a green suite from before the last edit looks exactly like a green suite
# after it.
NO_VERIFY_NOTE = (
    "GUARD: nothing has been run since your last edit, so nothing has "
    "demonstrated that the edit works -- and a suite that passed before it "
    "proves nothing about it. Run the tests that cover the code you changed "
    "with run_command, read what they actually said, and report that."
)


def _log() -> list[str]:
    """The ordered names of every tool call this run has made."""
    return [call.name for call in get_tool_calls()]


def write_count() -> int:
    """How many mutating calls this run has made, from the tool log.

    Asked of the log rather than of the model's report, because those are not
    the same claim and have been observed to disagree: a run that reports
    "Summary of changes made: modified django/db/models/fields/__init__.py"
    while having called only `grep` and `read_file` is not lying about a file it
    wrote -- it is describing the edit it decided on and never issued. Only the
    log knows.
    """
    return sum(1 for name in _log() if name in WRITE_TOOLS)


def ran_since_last_write() -> bool:
    """True if a command was run after the most recent write.

    Order is the whole of the question, which is why this reads the log rather
    than a pair of counters.
    """
    names = _log()
    last_write = max((i for i, n in enumerate(names) if n in WRITE_TOOLS), default=-1)
    if last_write < 0:
        return False
    return any(n in RUN_TOOLS for n in names[last_write + 1:])


def must_have_written(agent: "CodeAgent", result: Any, call: Any) -> None:
    """A report that claims a fix must be backed by a write in the log.

    A postcondition, so the failure lands back in the session the model is
    already in: NOOA routes `InvariantError` through its validation-retry
    feedback, which means the note arrives with the whole conversation still
    around it.

    A report that says the objective was not resolved is honest and allowed:
    the invariant is on claiming a fix, not on finishing.
    """
    if isinstance(result, Report) and not result.resolved:
        return
    if write_count() == 0:
        raise InvariantError(NO_WRITE_NOTE)


def must_have_verified(agent: "CodeAgent", result: Any, call: Any) -> None:
    """A report that claims a fix must have run something after the last edit.

    Deliberately "after the last edit" and not "at all". A repository's existing
    suite passes on an unfixed tree -- the test that would catch the bug is the
    one being held back -- so the evidence that fools a verifier is exactly the
    evidence it is looking at, and the only run that says anything about a
    change is one that happened after it.
    """
    if isinstance(result, Report) and not result.resolved:
        return
    if write_count() and not ran_since_last_write():
        raise InvariantError(NO_VERIFY_NOTE)


# Both generated methods take the same shape: a bounded session, and the two
# invariants above checked on whatever it returns.
WORK = CodeActConfig(
    max_iterations=CELLS_PER_STEP,
    postconditions=[must_have_written, must_have_verified],
)


class Report(BaseModel):
    """What one attempt concluded, and how to check it.

    A return contract in place of a router and a ledger. The other three sides
    build their answer by concatenating per-phase summaries, because each phase
    has its own conversation and the summary is the only thing that crosses;
    and they ask the model a separate question, constrained to the titles the
    topology allows, to learn whether it is done. Here one session does the work
    and its return type says what happened -- including the command that makes
    the claim checkable by someone other than the model.
    """

    resolved: bool = Field(
        description=(
            "True only if the objective is met and something you ran after your "
            "last edit demonstrated it. False if the change is incomplete, if "
            "the tests still fail, or if you ran out of room -- an honest False "
            "is worth more than a claim the evidence does not carry."
        )
    )
    summary: str = Field(
        description="What the root cause was, and how your change addresses it."
    )
    files_changed: list[str] = Field(
        description=(
            "Every repository-relative path you edited. Paths only, and nothing "
            "you did not write to."
        )
    )
    verify_command: str = Field(
        description=(
            "One run_command command line that passes if and only if your "
            "change is correct, for example 'pytest tests/test_thing.py'. It "
            "will be run again after you finish, so it must be a command this "
            "workspace accepts and it must exit non-zero if the fix is wrong. "
            "Leave it empty only if there is genuinely nothing to run."
        )
    )
    evidence: str = Field(
        description=(
            "What that command actually said when you ran it. Quote the lines "
            "that decided it."
        )
    )

    def rendered(self) -> str:
        """The report as the answer text the eval harness records."""
        files = "\n".join(f"- {p}" for p in self.files_changed) or "(none)"
        return (
            f"resolved: {self.resolved}\n\n"
            f"## Summary\n{self.summary}\n\n"
            f"## Files changed\n{files}\n\n"
            f"## Verified with\n{self.verify_command or '(nothing)'}\n\n"
            f"## Evidence\n{self.evidence}"
        )


class Checked(BaseModel):
    """The result of re-running a report's `verify_command` ourselves."""

    passed: bool
    output: str

    def rendered(self) -> str:
        state = "passed" if self.passed else "did NOT pass"
        return f"The verification command {state} when re-run:\n{self.output}"


class CodeAgent(Agent):
    """You are a coding agent working in one repository, on one objective.

    You work by writing Python in this session. The methods on `self` are how
    you reach the repository -- reading it, searching it, changing it, running
    things in it -- and each returns a string to read before deciding what to do
    next. Variables you define persist between cells, so you can hold what you
    found and build on it.

    How to work:

      * Look before you change anything. `grep` finds where a symbol lives far
        more cheaply than reading files does; then read only what it points at.
      * Copy text exactly when you use it as an edit anchor. `replace_in_file`
        matches literally, so an anchor retyped from memory will miss.
      * Fix the cause in the library source, not the symptom at the call site.
      * After you edit, run something. Tests that passed before your edit say
        nothing about it, and you may not report a fix you have not run.
      * Keep `set_plan` and `update_task` current on anything with more than a
        couple of steps: the plan is in your prompt every turn, and the rest of
        this conversation may be truncated.

    A result beginning "Error:" means try something different. One beginning
    "BLOCKED:" means that class of action is unavailable, so stop trying it.
    Neither is a reason to repeat the call that produced it.
    """

    # The plan is state, and it is the model's own: `state` renders it into
    # every turn, so what it wrote down survives a long stretch of tool traffic.
    plan: PlanTasks

    # The holders are hidden. They are the implementation of the methods below,
    # the methods are what the model is shown, and their internals include the
    # served-window memory -- which is not something a model should be able to
    # read its own way around.
    repo: Annotated[ExploreCodeBase, hidden]
    editor: Annotated[EditCode, hidden]
    runner: Annotated[VerifyCode, hidden]
    # Generated-method calls spent so far. Visible: one integer, and the thing
    # most worth knowing before deciding how much more to attempt.
    steps: int

    def __init__(self, repo_root: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Per instance, never at class level: all four own mutable state that
        # belongs to one run -- the served-window memory, the plan, the
        # execution backend's container binding.
        self.repo = ExploreCodeBase(repo_root=repo_root)
        self.editor = EditCode(repo_root=repo_root)
        self.runner = VerifyCode(repo_root=repo_root)
        self.plan = PlanTasks()
        self.steps = 0

    # -- reading the repository ---------------------------------------------

    def read_file(self, file_path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a text file from the repository.

        Returns the file's contents preceded by a header naming the path and the
        line range shown, as 'lines 12-340 of 900'. There are no line-number
        prefixes on the content, so text copied from here can be used directly
        as an edit anchor. A file small enough to fit comes back whole no matter
        what window you ask for, so read it once and work from that; a large one
        comes back one window at a time, and when the result ends by naming the
        next start_line, call read_file again with that start_line to continue.
        Repeating a call you have already made returns an error rather than the
        same text, so page forward with start_line or narrow the search with
        grep instead.

        Args:
            file_path: Path to the file, relative to the repository root.
            start_line: First line to show, 1-based. Use 1 to start at the top
                of the file.
            end_line: Last line to show, 1-based and inclusive. Use 0 to read to
                the end of the file.
        """
        return self.repo.read_file(file_path, start_line, end_line)

    def ls_repo(self, dir_path: str = ".") -> str:
        """List the immediate contents of one directory.

        Directory names are suffixed with '/'. This is not recursive -- use
        find_files to locate a file anywhere in the tree.

        Args:
            dir_path: Directory to list, relative to the repository root. Use
                '.' for the root itself.
        """
        return self.repo.ls_repo(dir_path)

    def find_files(self, name_glob: str, dir_path: str = ".") -> str:
        """Find files anywhere under a directory by matching their name against
        a glob pattern.

        Use this to locate a file when you know its name but not where it lives.

        Args:
            name_glob: Glob pattern matched against each file's name and its
                repository-relative path, for example '*.py' or 'tools/*.py'.
            dir_path: Directory to search under, relative to the repository
                root. Use '.' to search the whole repository.
        """
        return self.repo.find_files(name_glob, dir_path)

    def grep(self, pattern: str, path: str = ".", file_glob: str = "*") -> str:
        """Search file contents for a regular expression.

        Returns one line per match, formatted as 'path:line: text'. Use this to
        find where a symbol is defined or used before reading whole files.

        Args:
            pattern: Python regular expression to search for. Prefix it with
                (?i) for a case-insensitive search, for example '(?i)todo'.
            path: File or directory to search, relative to the repository root.
                A directory is searched recursively. Use '.' for the whole
                repository.
            file_glob: Glob restricting which file names are searched, for
                example '*.py'. Use '*' to search every file.
        """
        return self.repo.grep(pattern, path, file_glob)

    # -- changing it --------------------------------------------------------

    def write_file(self, file_path: str, content: str) -> str:
        """Create a new file, or replace an existing file's contents entirely.

        Missing parent directories are created automatically, and a trailing
        newline is added if the content lacks one. To change part of a file
        without re-sending the rest, use replace_in_file instead.

        Args:
            file_path: Path of the file to write, relative to the repository
                root.
            content: The complete new contents of the file.
        """
        return self.editor.write_file(file_path, content)

    def replace_in_file(
        self, file_path: str, old: str, new: str, expected_count: int = 1
    ) -> str:
        """Replace an exact piece of text in a file.

        The text is matched literally, not as a regular expression. By default
        it must occur exactly once, so include enough surrounding context to
        make it unique. If the text is not found, or occurs a different number
        of times than expected, the file is left unchanged and the error
        explains what to do next.

        Args:
            file_path: Path of the file to edit, relative to the repository
                root.
            old: The exact existing text to find, copied character for character
                from the file including its indentation.
            new: The text to put in its place.
            expected_count: How many occurrences of 'old' you expect to replace.
                Leave it at 1 to require a unique match; set it higher only to
                intentionally replace every occurrence.
        """
        return self.editor.replace_in_file(file_path, old, new, expected_count)

    # -- running things in it -----------------------------------------------

    def run_command(self, command: str, timeout_sec: int = 120) -> str:
        """Run one allowed command in the repository and return its exit code
        and output.

        Use 'jac check <path>' to type-check Jac code after editing it,
        'jac test <path>' or 'pytest' to run tests, and 'git status' or
        'git diff' to review changes. A non-zero exit code is a real result
        about the code, not a failure of this tool. The commands always accepted
        are: jac check, jac test, jac run, jac fmt, jac code, pytest,
        git status, git diff, git log, git show, git ls-files; some workspaces
        additionally allow an interpreter such as 'python', and a refusal always
        names the exact list this workspace accepts. There is no shell, so
        pipes, redirection, chaining with ';' or '&&', and command substitution
        do not work, and wildcards are NOT expanded -- write 'jac check tools/'
        to check a directory rather than 'jac check tools/*.py'.

        This is the only way to execute anything: the session blocks
        `subprocess` and `socket`, so a cell cannot spawn a process of its own.

        Args:
            command: The single command line to run, for example
                'pytest tests/'. Paths must be relative to the repository root.
            timeout_sec: How many seconds to allow before the command is killed.
                Use the default of 120 unless you expect a long test run.
        """
        return self.runner.run_command(command, timeout_sec)

    # -- keeping track ------------------------------------------------------

    def set_plan(self, steps: list[str]) -> str:
        """Replace the whole plan with a new ordered list of steps, numbered
        from 1.

        Call this once you know how you intend to solve the task, and again if
        the approach changes materially. Returns the rendered plan.

        Args:
            steps: The steps to take, in order, each a short imperative
                sentence.
        """
        return self.plan.set_plan(steps)

    def update_task(
        self,
        task_id: int,
        status: Literal["todo", "doing", "done", "blocked"],
        note: str = "",
    ) -> str:
        """Change the status of one step, optionally attaching a short note.

        Mark a step 'doing' when you start it and 'done' once you have verified
        it. Returns the rendered plan.

        Args:
            task_id: The number of the step to update, as shown in the plan.
            status: The step's new status.
            note: An optional short note, for example why a step is blocked.
                Pass an empty string to leave the existing note unchanged.
        """
        return self.plan.update_task(task_id, status, note)

    def show_plan(self) -> str:
        """Show the current plan with each step's status.

        Use this to re-orient after a long stretch of work.
        """
        return self.plan.show_plan()

    # -- what the model is asked to do --------------------------------------

    @strategy(CodeActStrategy(config=WORK))
    async def resolve(self, objective: str) -> Report:
        """Deliver this objective in the repository, then report what you did.

        Find the cause, change the source, run something that exercises the
        change, and return a Report. Set resolved only if what you ran after
        your last edit demonstrates the objective is met.

        Args:
            objective: What has to be true of the repository when you are done.
        """
        ...

    @strategy(CodeActStrategy(config=WORK))
    async def repair(self, objective: str, previous: Report, failure: str) -> Report:
        """Your last attempt did not hold up. Fix it and report again.

        `failure` is what happened when the verification command from your own
        report was re-run: read it before you touch anything. Then decide
        whether the change was wrong, incomplete, or in the wrong place, and fix
        that. The workspace still has your edits in it -- `git diff` shows them.

        If the objective genuinely cannot be met from here, return a Report with
        resolved set to False saying what is in the way. That is a result. A
        claim the evidence does not carry is not.

        Args:
            objective: What has to be true of the repository when you are done.
            previous: The report that did not hold up.
            failure: What re-running its verification command actually produced.
        """
        ...

    # -- the deterministic part ---------------------------------------------

    @hidden
    def check(self, report: Report) -> Checked:
        """Re-run the report's own verification command and read the exit code.

        This is the gate the other three sides of the comparison cannot have. A
        phase in a graph reports and the walker moves on; there is nothing left
        afterwards that could re-run what the phase claimed, so "Verifying said
        it passed" is as far as any of them gets. Here the claim arrives with a
        command attached, and the command is run again -- through the agent's own
        allowlisted `run_command`, so the gate cannot execute anything the model
        could not have executed itself, and the re-run is in the tool log like
        everything else.

        An empty command fails the gate. So does a refusal, a timeout, and any
        non-zero exit: `run_command` only reports "exit_code: 0" when a command
        actually spawned and actually succeeded.
        """
        command = (report.verify_command or "").strip()
        if not command:
            return Checked(
                passed=False,
                output=(
                    "No verification command was given, so the report cannot be "
                    "checked. Name one command that passes only if the change is "
                    "correct, and run it."
                ),
            )
        output = self.run_command(command)
        found = EXIT_CODE.search(output)
        return Checked(passed=bool(found) and found.group(1) == "0", output=output)

    @hidden
    async def run(self, objective: str, max_steps: int = DEFAULT_MAX_STEPS) -> Report:
        """Deliver the objective, and do not take the model's word for it.

        Hidden, so a generated cell cannot call the whole workflow back on
        itself. The control flow is ordinary Python -- one attempt, then a
        bounded repair loop over what the gate found -- and every guarantee the
        benchmark depends on lives here or in the invariants on the methods it
        calls, never in a question put to the model.

        A step is one generated-method call. `resolve` spends the first, each
        repair spends another, and the loop stops on a report that passes its
        own check, or on the budget. A report that honestly says it failed does
        not stop it: repairing that is what the budget is for.
        """
        self.steps += 1
        report = await self.resolve(objective)
        checked = self._gate(report)
        while not (report.resolved and checked.passed) and self.steps < max_steps:
            self.steps += 1
            report = await self.repair(objective, report, checked.rendered())
            checked = self._gate(report)
        # The last word on `resolved` is the gate's, not the report's: what the
        # harness records must not claim more than was demonstrated.
        if report.resolved and not checked.passed:
            report = report.model_copy(
                update={
                    "resolved": False,
                    "evidence": f"{report.evidence}\n\n(re-checked: {checked.output})",
                }
            )
        return report

    @hidden
    def _gate(self, report: Report) -> Checked:
        """One check per report, and none wasted.

        A report that admits it failed has already said more than the gate could,
        and re-running a command that was never going to pass costs a full test
        suite for nothing -- so its own account is what goes back to `repair`.
        """
        if not report.resolved:
            return Checked(
                passed=False,
                output="Your own report set resolved to False:\n" + report.evidence,
            )
        return self.check(report)


__all__ = [
    "CELLS_PER_STEP",
    "DEFAULT_MAX_STEPS",
    "NO_VERIFY_NOTE",
    "NO_WRITE_NOTE",
    "RUN_TOOLS",
    "WORK",
    "WRITE_TOOLS",
    "Checked",
    "CodeAgent",
    "Report",
    "must_have_verified",
    "must_have_written",
    "ran_since_last_write",
    "write_count",
]
