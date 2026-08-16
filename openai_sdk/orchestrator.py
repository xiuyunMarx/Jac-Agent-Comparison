"""The coding agent: a phase graph walked by a plain loop.

Five phases, each granted exactly the tools it is permitted to call. One run
state walks that graph; at each phase the model is handed that phase's tools and
left to work until it reports done. The per-phase ReAct loop lives in
phase_agent.py; the tools themselves live in tools/.

The capability boundary and the topology are both kept as data rather than as
control flow:

  * "which tools may this phase use" is `Phase.exposes`, a tuple of `ToolSpec`,
    so the boundary can be inspected and asserted instead of being a dictionary
    buried in a closure. byLLM spells it as `edge Exposes: Phase --> Tool`.
  * "where do I go next" is `FLOW`, declared once with the reason for every
    edge, so the topology -- the Verifying -> Editing repair loop, and the
    read-only shortcuts to Finished -- lives in one place. byLLM spells it as
    `edge Flow: Phase --> Phase`, and the reasons are what its router reads.

This is the no-framework side of a three-way comparison. The other two are
../byLLM/orchestrator.jac (Jac + byLLM) and ../langgraph/orchestrator.py
(Python + LangGraph). Where those two have drifted apart, this side follows
byLLM; README.md records where and why.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from llm import (
    DEFAULT_MODEL,
    active_model_name,
    complete,
    set_client,
    set_model,
    token_usage,
)
from llm_log import log_llm_calls
from phase_agent import run_phase
from tools.common import ToolCall, get_tool_calls, reset_tool_log
from tools.edit import EditCode
from tools.explore import ExploreCodeBase
from tools.plan import PlanTasks
from tools.spec import Tool
from tools.verify import VerifyCode

# Re-exported so the SWE-bench shim -- which imports `solve`,
# `active_model_name` and `DEFAULT_MODEL` from whichever `orchestrator` module
# $CODEAGENT_HOME resolves to -- reads the same names on all three sides.
__all__ = [
    "DEFAULT_MODEL",
    "FLOW",
    "AgentRuntime",
    "Phase",
    "RunResult",
    "active_model_name",
    "build_pipeline",
    "main",
    "set_client",
    "set_model",
    "solve",
]

# The only two tools that change the workspace. Every other call, however many
# of them a phase makes, leaves the tree exactly as it was found.
WRITE_TOOLS = ("write_file", "replace_in_file")

# What Verifying is told when it is sent back for having verified nothing. It
# has to name the specific omission: a phase that is only told "try again"
# repeats itself, because from inside the conversation it already believes the
# edit was made.
WRITE_GUARD_NOTE = (
    "GUARD: no write_file or replace_in_file call has been made in this run, "
    "so the workspace is still exactly as it was found and the commands above "
    "prove nothing about the objective. Any change described so far was "
    "planned, not applied. Go back to Editing and issue the edit."
)


def workspace_written() -> bool:
    """True once this run has actually modified a file in the workspace.

    Asked of the tool log rather than of the phase summary, because those are
    not the same claim and have been observed to disagree. A phase that reports
    "Summary of changes made: modified django/db/models/fields/__init__.py"
    while having called only `grep` and `read_file` is not lying about a file it
    wrote -- it is describing the edit it decided on and never issued. Only the
    log knows.
    """
    return any(call.name in WRITE_TOOLS for call in get_tool_calls())


# ---------------------------------------------------------------------------
# Runtime: owns the four tool holders and the repository they are pinned to.
# ---------------------------------------------------------------------------


class AgentRuntime:
    """byLLM's AgentRuntime, minus `phase_tools` and `phase_ctx`.

    Those two exist on that side only because byLLM serializes a `by` function's
    parameters into the prompt, so passing the tool list or the conversation as
    arguments would dump them into every request as object reprs. Here both are
    ordinary arguments to `run_phase`, so the smuggling is not needed.
    """

    def __init__(self, repo_root: str = "") -> None:
        self.repo_root: str = ""
        self.explorer: ExploreCodeBase
        self.editor: EditCode
        self.planner: PlanTasks
        self.verifier: VerifyCode
        self.retarget(repo_root)

    def retarget(self, path: str) -> None:
        base = os.path.realpath(path or os.getcwd())
        self.repo_root = base
        self.explorer = ExploreCodeBase(repo_root=base)
        self.editor = EditCode(repo_root=base)
        self.planner = PlanTasks()
        self.verifier = VerifyCode(repo_root=base)


rt = AgentRuntime()


# ---------------------------------------------------------------------------
# Phases: the capability boundary and the topology, both as data.
# ---------------------------------------------------------------------------


class ToolCat(str, Enum):
    READ = "read"
    WRITE = "write"
    RUN = "run"
    TRACK = "track"


@dataclass(frozen=True)
class ToolSpec:
    """One capability a phase is allowed to call. byLLM's `node Tool`."""

    tool: Tool
    category: str = ToolCat.READ.value


@dataclass(frozen=True)
class Phase:
    title: str
    goal: str = ""
    effect: str = ""
    exposes: tuple[ToolSpec, ...] = ()

    def tools(self) -> list[Tool]:
        return [spec.tool for spec in self.exposes]

    def tool_names(self) -> list[str]:
        return sorted(spec.tool.name for spec in self.exposes)

    def categories(self) -> list[str]:
        return sorted({spec.category for spec in self.exposes})


@dataclass(frozen=True)
class FlowEdge:
    """One edge out of a phase, and why a run would take it.

    `reason` is not decoration: it is what the router is shown for each
    candidate, and it is byLLM's `edge Flow { has reason: str }` verbatim.
    """

    target: str
    reason: str


FINISHED = "Finished"
WORKING_PHASES: tuple[str, ...] = ("Planning", "Exploring", "Editing", "Verifying")

PLANNING_GOAL = (
    "Understand what is being asked and write down the steps you will take. "
    "Call set_plan with those steps before you finish."
)
EXPLORING_GOAL = (
    "Locate the exact files and lines that must change. Report the paths and "
    "what each one needs."
)
EDITING_GOAL = (
    "Make the changes. Use write_file for new files and replace_in_file for "
    "surgical edits. Actually call the tools -- describing a change is not "
    "making it."
)
VERIFYING_GOAL = (
    "Prove the change works by running it. Use run_command with jac check, "
    "jac test or pytest, and report exactly what the output showed."
)
# Never handed to a phase visit -- Finished runs no phase -- so this exists
# purely to describe the node to the router, which would otherwise see the
# terminal choice as the emptiest candidate on the list.
FINISHED_GOAL = "Stop. The objective is done and verified."

PLANNING_EFFECT = "Spends one step, then routes on."
EXPLORING_EFFECT = "Spends one step reading the repository. Cannot write."
EDITING_EFFECT = "Spends one step and writes to disk. Always routes to Verifying afterwards."
VERIFYING_EFFECT = "Spends one step running commands. Routes back to Editing if anything is wrong."
FINISHED_EFFECT = (
    "Ends the run. The ledger becomes the final answer; nothing runs after this."
)

# The phase topology, declared once. A router may only return a target listed
# here for that phase.
FLOW: dict[str, tuple[FlowEdge, ...]] = {
    "Planning": (
        FlowEdge("Exploring", "the plan is written; go find the code it touches"),
        FlowEdge(FINISHED, "shortcut: the request was a question the plan already answers"),
    ),
    "Exploring": (
        FlowEdge("Editing", "the code to change has been located; go change it"),
        FlowEdge(FINISHED, "shortcut: the request was a question the code just read answers"),
    ),
    "Editing": (
        FlowEdge("Verifying", "an edit was made and must be proven by running it"),
    ),
    "Verifying": (
        FlowEdge("Editing", "the repair loop: verification found problems left to fix"),
        FlowEdge(FINISHED, "verification proved the objective complete and correct"),
    ),
    FINISHED: (),
}

# What the router is told to weigh, per phase. Copied from the `intent=` strings
# on byLLM's `visit [edge ->:Flow:->] by agent_model(select=1, ...)` clauses.
ROUTER_INTENT: dict[str, str] = {
    "Planning": (
        "Go to Exploring to find the code this task touches. Choose Finished "
        "only if this was a question needing no code change and the plan "
        "already answers it."
    ),
    "Exploring": (
        "Go to Editing to make the change this exploration located. Choose "
        "Finished only if the objective was a question that the code just read "
        "already answers, with nothing left to edit."
    ),
    "Verifying": (
        "Choose ```Finished``` node only if the verification run demonstrated "
        "the objective is complete and correct -- not merely that an attempt "
        "was made. If it failed, was skipped, or is ambiguous, go back to "
        "Editing."
    ),
}


def expose(tools: Sequence[Tool], category: ToolCat) -> tuple[ToolSpec, ...]:
    return tuple(ToolSpec(tool=tool, category=category.value) for tool in tools)


def build_pipeline(runtime: AgentRuntime | None = None) -> dict[str, Phase]:
    """Grant each phase its tools.

    Every phase can read and can update the plan; only Editing can write, and
    only Editing and Verifying can execute.
    """
    runtime = runtime or rt
    read_tools = runtime.explorer.as_tools()
    track_tools = [runtime.planner.show_plan_tool(), runtime.planner.update_task_tool()]
    write_tools = runtime.editor.as_tools()
    run_tools = runtime.verifier.as_tools()

    common = expose(read_tools, ToolCat.READ) + expose(track_tools, ToolCat.TRACK)
    phases = [
        Phase(
            title="Planning",
            goal=PLANNING_GOAL,
            effect=PLANNING_EFFECT,
            exposes=common + expose([runtime.planner.set_plan_tool()], ToolCat.TRACK),
        ),
        Phase(
            title="Exploring",
            goal=EXPLORING_GOAL,
            effect=EXPLORING_EFFECT,
            exposes=common,
        ),
        Phase(
            title="Editing",
            goal=EDITING_GOAL,
            effect=EDITING_EFFECT,
            exposes=common
            + expose(write_tools, ToolCat.WRITE)
            + expose(run_tools, ToolCat.RUN),
        ),
        Phase(
            title="Verifying",
            goal=VERIFYING_GOAL,
            effect=VERIFYING_EFFECT,
            exposes=common + expose(run_tools, ToolCat.RUN),
        ),
        Phase(title=FINISHED, goal=FINISHED_GOAL, effect=FINISHED_EFFECT),
    ]
    return {phase.title: phase for phase in phases}


# ---------------------------------------------------------------------------
# Run state. byLLM's `walker CodeAgent`, which carries exactly these fields.
# ---------------------------------------------------------------------------


@dataclass
class AgentRun:
    objective: str
    max_steps: int = 10
    steps: int = 0
    ledger: list[str] = field(default_factory=list)
    answer: str = ""
    edits_made: bool = False

    def record(self, phase: str, summary: str) -> None:
        self.ledger.append(f"### {phase}\n{summary}")

    def rendered(self) -> str:
        return "\n\n".join(self.ledger)

    def exhausted(self) -> bool:
        """True when the step budget is spent, so every phase stops advancing."""
        return self.steps >= self.max_steps


def phase_system_prompt(phase: Phase, repo_root: str, run: AgentRun) -> str:
    return (
        f"You are a coding agent working in the repository at {repo_root}.\n"
        f"You are in the {phase.title} phase: {phase.goal}\n\n"
        f"## Objective\n{run.objective}\n\n"
        f"## Progress so far\n{run.rendered() or '(nothing yet)'}\n\n"
        "Use the tools you have been given. Read before you edit, and copy "
        "text exactly when you use it as an edit anchor. When the phase's "
        "goal is met, stop and summarise."
    )


# ---------------------------------------------------------------------------
# The router: byLLM's `visit [edge ->:Flow:->] by agent_model(select=1, ...)`.
# ---------------------------------------------------------------------------

ROUTER_SYSTEM = (
    "You are routing a coding agent between the phases of one run. You will be "
    "given the phases it may move to next, why each edge exists, what the agent "
    "would do there, and what it has done so far. Choose exactly one."
)


def router_schema(titles: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "next_phase",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"next": {"type": "string", "enum": list(titles)}},
                "required": ["next"],
                "additionalProperties": False,
            },
        },
    }


def render_candidates(
    edges: Sequence[FlowEdge], phases: Mapping[str, Phase]
) -> str:
    lines: list[str] = []
    for edge in edges:
        target = phases[edge.target]
        lines.append(
            f"- {edge.target}\n"
            f"    take this edge when: {edge.reason}\n"
            f"    what happens there: {target.goal}\n"
            f"    effect: {target.effect}"
        )
    return "\n".join(lines)


def read_choice(text: str, titles: Sequence[str]) -> str:
    """The chosen title out of a model reply, or "" if it named none of them."""
    if not text:
        return ""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            picked = str(parsed.get("next", ""))
            if picked in titles:
                return picked
    except ValueError:
        pass
    # A model that ignored the schema still usually says the name. Accept it
    # only when exactly one candidate appears, so "not Finished, go to Editing"
    # is a miss rather than a coin toss.
    named = [t for t in titles if t.lower() in text.lower()]
    return named[0] if len(named) == 1 else ""


def select_edge(
    source: str,
    edges: Sequence[FlowEdge],
    phases: Mapping[str, Phase],
    run: AgentRun,
    *,
    usage: Any = None,
    client: Any = None,
    model: str | None = None,
) -> str:
    """Ask the model which edge to take. Returns a target title.

    On any failure -- a provider that rejects strict schemas, a reply naming no
    candidate, an exception -- this falls back to the FIRST edge declared for
    the phase. For Verifying that is Editing, which is the conservative
    direction: calling the objective done without evidence is the expensive
    error, and the step budget bounds the repair loop either way.
    """
    titles = [edge.target for edge in edges]
    prompt = (
        f"## Intent\n{ROUTER_INTENT.get(source, '')}\n\n"
        f"## Objective\n{run.objective}\n\n"
        f"## Candidates\n{render_candidates(edges, phases)}\n\n"
        f"## Progress so far\n{run.rendered() or '(nothing yet)'}"
    )
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    try:
        message = complete(
            messages,
            response_format=router_schema(titles),
            usage=usage,
            client=client,
            model=model,
        )
        picked = read_choice(str(getattr(message, "content", "") or ""), titles)
        if picked:
            return picked
        # One retry in plain text: some providers and proxies reject
        # `response_format` outright and the error is indistinguishable from a
        # model that simply did not answer in JSON.
        message = complete(
            messages
            + [{"role": "user", "content": f"Answer with exactly one of: {', '.join(titles)}."}],
            usage=usage,
            client=client,
            model=model,
        )
        picked = read_choice(str(getattr(message, "content", "") or ""), titles)
        if picked:
            return picked
    except Exception as e:  # noqa: BLE001 - routing must not strand the run
        sys.stderr.write(f"[router] {source}: {type(e).__name__}: {e}\n")
    return titles[0]


# ---------------------------------------------------------------------------
# The walk. byLLM spawns a walker over the graph; here the graph is walked.
# ---------------------------------------------------------------------------


def walk(
    run: AgentRun,
    phases: Mapping[str, Phase],
    runtime: AgentRuntime,
    *,
    usage: Any = None,
    client: Any = None,
    model: str | None = None,
) -> AgentRun:
    title = "Planning"
    while title != FINISHED:
        phase = phases[title]
        # The explorer's served-window memory is per phase, because each phase
        # keeps its own conversation: a window Exploring read is genuinely
        # absent from Editing's messages, so Editing must still be served it.
        runtime.explorer.set_scope(phase.title)
        # Only the system message crosses into the next visit of a phase.
        # Everything after it would be the previous lap's tool traffic -- file
        # dumps and test output the phase has already summarised, and that
        # summary is in the ledger this very message carries. Kept, it made
        # Editing's fourth lap arrive with four full pytest runs in tow, 22k of
        # its 28k prompt tokens, all describing a file it had since edited three
        # more times. (This is the one place langgraph/ diverged from byLLM; the
        # byLLM behaviour is the one reproduced here.)
        conversation = [
            {
                "role": "system",
                "content": phase_system_prompt(phase, runtime.repo_root, run),
            }
        ]
        result = run_phase(
            phase.goal,
            phase.tools(),
            conversation,
            usage=usage,
            client=client,
            model=model,
        )
        run.steps += 1
        run.record(phase.title, result.summary)
        if phase.title == "Editing":
            # Asked of the tool log, not assumed from having run. This used to
            # be an unconditional True, which recorded "Editing was visited"
            # under a name that reads as "an edit was made" -- and nothing ever
            # read it, so the untruth stayed invisible until a run reported a
            # fix it had never written.
            run.edits_made = workspace_written()

        edges = FLOW[phase.title]
        # The budget is not a judgement call, so it is settled before the model
        # is asked: a spent budget is exactly when another LLM call is the wrong
        # move.
        if run.exhausted():
            title = FINISHED
        elif phase.title == "Verifying" and not run.edits_made:
            # Nothing was written, so nothing was verified -- whatever the run
            # just observed, it observed about the unmodified tree. Settled
            # before select_edge is asked, not by it, because the evidence that
            # fools it is exactly the evidence it is looking at: a repository's
            # existing suite passes on an unfixed tree, since the test that
            # would catch the bug is the one being held back. Green tests here
            # mean "I changed nothing", and the router reads them as proof.
            run.record(phase.title, WRITE_GUARD_NOTE)
            title = "Editing"
        elif len(edges) == 1:
            title = edges[0].target
        else:
            title = select_edge(
                phase.title, edges, phases, run, usage=usage, client=client, model=model
            )
    run.answer = run.rendered()
    return run


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """What one agent run produced.

    Typed rather than a dict, with the same fields as byLLM's and LangGraph's
    RunResult, so the eval harness reads a stable shape on all three sides.
    """

    objective: str
    answer: str = ""
    steps: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def solve(objective: str, repo_root: str = "", max_steps: int = 10) -> RunResult:
    rt.retarget(repo_root)
    reset_tool_log()
    token_usage.reset()
    run = walk(AgentRun(objective=objective, max_steps=max_steps), build_pipeline(), rt)
    totals = token_usage.totals()
    return RunResult(
        objective=objective,
        answer=run.answer,
        steps=run.steps,
        tool_calls=get_tool_calls(),
        llm_calls=totals["llm_calls"],
        prompt_tokens=totals["prompt_tokens"],
        completion_tokens=totals["completion_tokens"],
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the openai-sdk orchestrator."
    )
    parser.add_argument("--task", required=True, help="The task to be solved.")
    parser.add_argument(
        "--repo-path",
        default="",
        help="Path to the repository to work in. Defaults to the current directory.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="How many phase visits the run may spend.",
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    outcome = solve(args.task, args.repo_path, args.max_steps)
    # stdout is the answer channel -- the eval harness reads it -- so the
    # telemetry line and the log note both go to stderr.
    print(outcome.answer)
    sys.stderr.write(
        f"[{outcome.steps} phases, {len(outcome.tool_calls)} tool calls, "
        f"{outcome.llm_calls} llm calls, "
        f"{outcome.prompt_tokens}+{outcome.completion_tokens} tokens]\n"
    )
    log_llm_calls()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
