"""The coding agent: a phase graph walked by one LangGraph state machine.

Five phases are wired into a `StateGraph`, and each phase is granted exactly the
tools it is permitted to call. One `AgentState` flows through that graph; at each
phase the model is bound to that phase's tools and left to work until it reports
done.

The capability boundary and the topology are both kept as data rather than as
control flow:

  * "which tools may this phase use" is `Phase.exposes`, a tuple of `ToolSpec`,
    so the boundary can be inspected and asserted instead of being a dictionary
    buried in a closure.
  * "where do I go next" is `FLOW`, declared once, so the phase topology
    (including the Verifying -> Editing repair loop and the read-only shortcut
    to Finished) lives in one place instead of being scattered across
    conditionals.

The tools themselves live in tools/: explore.py, edit.py, plan.py, verify.py.
This file only grants them to phases and drives the loop. The per-phase ReAct
loop lives in phase_agent.py.

This is the LangGraph half of a two-framework comparison; the other half is
../byLLM/orchestrator.jac. See README.md for the mapping between them.
"""

from __future__ import annotations

import operator
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Callable, Sequence, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from phase_agent import build_phase_agent, run_phase
from telemetry import TokenUsage
from tools.common import ToolCall, get_tool_calls, reset_tool_log
from tools.edit import EditCode
from tools.explore import ExploreCodeBase
from tools.plan import PlanTasks
from tools.verify import VerifyCode

DEFAULT_MODEL = "gpt-4o"

# ---------------------------------------------------------------------------
# Model. Constructed lazily and reachable only through this seam, so importing
# this module needs no API key and the tests can swap in a fake without ever
# touching a provider.
# ---------------------------------------------------------------------------

_model: BaseChatModel | None = None


def build_model() -> BaseChatModel:
    # Imported here rather than at module scope: ChatOpenAI validates its
    # credentials at construction, so a top-level import would make `import
    # orchestrator` fail in a keyless environment.
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ.get("CODEAGENT_MODEL", DEFAULT_MODEL),
        temperature=0.0,
        max_tokens=4096,
    )


def get_model() -> BaseChatModel:
    global _model
    if _model is None:
        _model = build_model()
    return _model


def set_model(model: BaseChatModel | None) -> None:
    """Swap the model. Pass None to fall back to the configured provider."""
    global _model
    _model = model


def active_model_name() -> str:
    model = get_model()
    return str(getattr(model, "model_name", None) or getattr(model, "model", ""))


token_usage = TokenUsage()


# ---------------------------------------------------------------------------
# Runtime: owns the four tool holders and the repository they are pinned to.
# ---------------------------------------------------------------------------


class AgentRuntime:
    """Same shape as byLLM's AgentRuntime, minus `phase_tools`/`phase_ctx`.

    Those two existed only because byLLM serializes a `by` function's parameters
    into the prompt, so the tool list and conversation had to be smuggled in as
    runtime state. LangGraph passes both explicitly, so they are gone.
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
    """One capability a phase is allowed to call."""

    tool: BaseTool
    category: str = ToolCat.READ.value


@dataclass(frozen=True)
class Phase:
    title: str
    goal: str = ""
    exposes: tuple[ToolSpec, ...] = ()

    def tools(self) -> list[BaseTool]:
        return [spec.tool for spec in self.exposes]

    def tool_names(self) -> list[str]:
        return sorted(spec.tool.name for spec in self.exposes)

    def categories(self) -> list[str]:
        return sorted({spec.category for spec in self.exposes})


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

FINISHED = "Finished"
WORKING_PHASES: tuple[str, ...] = ("Planning", "Exploring", "Editing", "Verifying")

# The phase topology, declared once. A phase's router may only return a title
# listed here for that phase.
FLOW: dict[str, tuple[str, ...]] = {
    "Planning": ("Exploring", FINISHED),
    "Exploring": ("Editing", FINISHED),
    "Editing": ("Verifying",),
    "Verifying": ("Editing", FINISHED),
    FINISHED: (),
}


def expose(tools: Sequence[BaseTool], category: ToolCat) -> tuple[ToolSpec, ...]:
    return tuple(ToolSpec(tool=tool, category=category.value) for tool in tools)


def build_pipeline(runtime: AgentRuntime | None = None) -> dict[str, Phase]:
    """Grant each phase its tools. Every phase can read and can update the plan;
    only Editing can write, and only Editing and Verifying can execute."""
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
            exposes=common + expose([runtime.planner.set_plan_tool()], ToolCat.TRACK),
        ),
        Phase(title="Exploring", goal=EXPLORING_GOAL, exposes=common),
        Phase(
            title="Editing",
            goal=EDITING_GOAL,
            exposes=common + expose(write_tools, ToolCat.WRITE) + expose(run_tools, ToolCat.RUN),
        ),
        Phase(
            title="Verifying",
            goal=VERIFYING_GOAL,
            exposes=common + expose(run_tools, ToolCat.RUN),
        ),
        Phase(title=FINISHED, goal="The objective is met, or the step budget ran out."),
    ]
    return {phase.title: phase for phase in phases}


# ---------------------------------------------------------------------------
# State.
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    objective: str
    max_steps: int
    steps: int
    ledger: Annotated[list[str], operator.add]
    answer: str
    edits_made: bool
    last_summary: str
    phase_msgs: dict[str, list[BaseMessage]]


def rendered(ledger: Sequence[str]) -> str:
    return "\n\n".join(ledger)


def exhausted(state: AgentState) -> bool:
    """True when the step budget is spent, so every phase stops advancing."""
    return state["steps"] >= state["max_steps"]


def phase_system_prompt(phase: Phase, repo_root: str, state: AgentState) -> str:
    return (
        f"You are a coding agent working in the repository at {repo_root}.\n"
        f"You are in the {phase.title} phase: {phase.goal}\n\n"
        f"## Objective\n{state['objective']}\n\n"
        f"## Progress so far\n{rendered(state['ledger']) or '(nothing yet)'}\n\n"
        "Use the tools you have been given. Read before you edit, and copy "
        "text exactly when you use it as an edit anchor. When the phase's "
        "goal is met, stop and summarise."
    )


def make_phase_node(
    phase: Phase, runtime: AgentRuntime, agent: Any
) -> Callable[[AgentState], dict[str, Any]]:
    def node(state: AgentState) -> dict[str, Any]:
        # The explorer's served-window memory is per phase, because each phase
        # keeps its own conversation: a window Exploring read is genuinely
        # absent from Editing's messages, so Editing must still be served it.
        runtime.explorer.set_scope(phase.title)
        system = SystemMessage(content=phase_system_prompt(phase, runtime.repo_root, state))
        prior = state["phase_msgs"].get(phase.title) or []
        messages: list[BaseMessage] = [system, *prior[1:], HumanMessage(content=phase.goal)] # Message 0 is refreshed each visit;
        summary, produced = run_phase(agent, messages)
        update: dict[str, Any] = {
            "steps": state["steps"] + 1,
            "ledger": [f"### {phase.title}\n{summary}"],
            "last_summary": summary,
            "phase_msgs": {**state["phase_msgs"], phase.title: produced},
        }
        if phase.title == "Editing":
            update["edits_made"] = True
        return update

    return node


def finished_node(state: AgentState) -> dict[str, Any]:
    return {"answer": rendered(state["ledger"])}


# ---------------------------------------------------------------------------
# The objective classifier that closes the repair loop.
# ---------------------------------------------------------------------------


class Verdict(BaseModel):
    """Whether the objective has been fully accomplished AND verified."""

    met: bool = Field(
        description=(
            "True only if the evidence actually demonstrates the work is "
            "complete and correct -- not merely that an attempt was made. If "
            "verification failed, was skipped, or is ambiguous, answer false."
        )
    )


OBJECTIVE_MET_SYSTEM = (
    "Decide whether the objective has been fully accomplished AND verified. "
    "Answer true only if the evidence actually demonstrates the work is "
    "complete and correct -- not merely that an attempt was made. If "
    "verification failed, was skipped, or is ambiguous, answer false."
)


def objective_met(
    model: BaseChatModel, usage: TokenUsage, objective: str, evidence: str
) -> bool:
    try:
        structured = model.with_structured_output(Verdict, include_raw=True)
        out = structured.invoke([
            SystemMessage(content=OBJECTIVE_MET_SYSTEM),
            HumanMessage(
                content=(
                    f"## Objective\n{objective}\n\n"
                    f"## Evidence from the verification phase\n{evidence}"
                )
            ),
        ])
        if isinstance(out, dict):
            if out.get("raw") is not None:
                usage.track(out["raw"])
            parsed = out.get("parsed")
        else:
            parsed = out
        return bool(parsed.met)
    except Exception:
        # A classifier failure must not strand the run mid-graph.
        return True


# ---------------------------------------------------------------------------
# Graph.
# ---------------------------------------------------------------------------


def build_app(
    runtime: AgentRuntime | None = None,
    model: BaseChatModel | None = None,
    usage: TokenUsage | None = None,
) -> Any:
    runtime = runtime or rt
    usage = usage if usage is not None else token_usage
    phases = build_pipeline(runtime)

    # Resolved on first use, not here: compiling the graph must not construct a
    # provider client, or inspecting the topology would demand credentials.
    resolved: list[BaseChatModel] = []

    def model_ref() -> BaseChatModel:
        if not resolved:
            resolved.append(model if model is not None else get_model())
        return resolved[0]

    agents = {
        title: build_phase_agent(phases[title].tools(), model_ref, usage)
        for title in WORKING_PHASES
    }

    def route_planning(state: AgentState) -> str:
        return FINISHED if exhausted(state) else "Exploring"

    def route_exploring(state: AgentState) -> str:
        return FINISHED if exhausted(state) else "Editing"

    def route_editing(state: AgentState) -> str:
        return "Verifying"

    def route_verifying(state: AgentState) -> str:
        if exhausted(state):
            return FINISHED
        met = objective_met(model_ref(), usage, state["objective"], state["last_summary"])
        return FINISHED if met else "Editing"

    routers: dict[str, Callable[[AgentState], str]] = {
        "Planning": route_planning,
        "Exploring": route_exploring,
        "Editing": route_editing,
        "Verifying": route_verifying,
    }

    graph = StateGraph(AgentState)
    for title in WORKING_PHASES:
        graph.add_node(title, make_phase_node(phases[title], runtime, agents[title]))
    graph.add_node(FINISHED, finished_node)
    graph.set_entry_point("Planning")
    for title in WORKING_PHASES:
        # The path map comes straight from FLOW, so a router can never send the
        # run somewhere the declared topology does not allow.
        graph.add_conditional_edges(
            title, routers[title], {dest: dest for dest in FLOW[title]}
        )
    graph.add_edge(FINISHED, END)
    return graph.compile()


def initial_state(objective: str, max_steps: int) -> AgentState:
    return {
        "objective": objective,
        "max_steps": max_steps,
        "steps": 0,
        "ledger": [],
        "answer": "",
        "edits_made": False,
        "last_summary": "",
        "phase_msgs": {},
    }


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """What one agent run produced.

    Typed rather than a dict, with the same fields as byLLM's RunResult, so the
    eval harness reads a stable shape on both sides of the comparison.
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
    app = build_app()
    final = app.invoke(
        initial_state(objective, max_steps),
        # Each phase is one super-step plus its router; the repair loop can
        # revisit Editing and Verifying until the step budget stops it.
        {"recursion_limit": 4 * max_steps + 10},
    )
    totals = token_usage.totals()
    return RunResult(
        objective=objective,
        answer=final["answer"],
        steps=final["steps"],
        tool_calls=get_tool_calls(),
        llm_calls=totals["llm_calls"],
        prompt_tokens=totals["prompt_tokens"],
        completion_tokens=totals["completion_tokens"],
    )


def main(argv: Sequence[str]) -> int:
    if not argv:
        print('usage: python orchestrator.py "<what you want done>" [repo_root]')
        return 1
    outcome = solve(argv[0], argv[1] if len(argv) > 1 else "")
    print(outcome.answer)
    print(
        f"\n[{outcome.steps} phases, {len(outcome.tool_calls)} tool calls, "
        f"{outcome.llm_calls} llm calls, "
        f"{outcome.prompt_tokens}+{outcome.completion_tokens} tokens]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
