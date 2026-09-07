"""Five-phase coding workflow using LangGraph and standard LangChain agents."""

from __future__ import annotations

import sys
from dataclasses import asdict
from typing import Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from pydantic import ValidationError, create_model

from nodes import (PHASES, TOOL_BUDGET, Phase, PhaseCapability, edges_from, get_model,
                   history, ran_since_edit, repo, tool_log, written)

MAX_REPAIRS: int = 3
MAX_RELOCATES: int = 1

SYSTEM: str = (
    "You are a coding agent fixing an issue in the repository at the current directory.\n"
    "Edit only library source, never tests. Read before you edit; copy anchor text exactly.\n"
    "Work in phases; each phase tells you its goal and gives you only the tools it needs.\n\n"
    "## Issue\n{issue}"
)

REPAIR_INTENT = (
    "Pick the edge whose capability matches the state of the work. Take the 'finish' edge only "
    "if the repository's own tests covering the changed code ran after the last edit and passed, "
    "and the reproduction now passes. Otherwise take the 'repair' edge."
)
RELOCATE_INTENT = (
    "Pick the edge whose capability matches the evidence. 'finish' only if the reproduction and "
    "the covering tests pass after the last edit. 'relocate' if the reproduction still fails the "
    "same way it did before any edit. 'repair' if the edit changed the behaviour but not enough."
)


FINAL_INSTRUCTION = "Based on the tool calls and their results above, provide only your final answer."


@wrap_tool_call
def tool_errors(request, handler):
    """Keep tool failures recoverable, as in the other implementations."""
    try:
        return handler(request)
    except Exception as error:
        return ToolMessage(content=f"Error: {request.tool_call['name']} failed: {type(error).__name__}: {error}",
                           tool_call_id=request.tool_call["id"])


def work(phase: Phase) -> str:
    rounds = 0

    @wrap_model_call
    def budget(request, handler):
        nonlocal rounds
        # The shared budget is checked between rounds, after the first call.
        if rounds and (0 < phase.max_react_iterations <= rounds or len(tool_log) >= TOOL_BUDGET):
            request = request.override(tools=[], messages=[*request.messages, HumanMessage(FINAL_INSTRUCTION)])
        rounds += 1
        return handler(request)

    agent = create_agent(get_model(), tools=phase.toolbox(), middleware=[budget, tool_errors])
    result = agent.invoke(
        {"messages": [*history, HumanMessage(f"{phase.sem}\n\n{phase.goal}")]},
        {"max_concurrency": 1, "recursion_limit": 2 * phase.max_react_iterations + 10},
    )
    history[:] = result["messages"]
    return history[-1].text.strip()


def select_edge(candidates: list[PhaseCapability], intent: str, incl_info: dict,
                walker: dict, here: str) -> PhaseCapability | None:
    prompt = (f"{intent}\n\nAgent state: {walker}\nCurrent phase: {PHASES[here].goal}\n"
              f"Candidates: {[asdict(edge) for edge in candidates]}\nContext: {incl_info}\n"
              "Choose one of the candidate targets.")
    schema = create_model("Route", target=(Literal[tuple(edge.target for edge in candidates)], ...))
    router = get_model().with_structured_output(schema, method="function_calling").with_retry(
        retry_if_exception_type=(OutputParserException, ValidationError),
        stop_after_attempt=2, wait_exponential_jitter=False,
    )
    try:
        choice = router.invoke(prompt)
        return next((edge for edge in candidates if edge.target == choice.target), None)
    except Exception:
        # An unavailable or invalid routing answer falls back to Edit in main.
        return None


class AgentState(TypedDict):
    """Run state and the next phase selected by Verify."""

    issue: str
    repo_root: str
    ledger: list[str]
    repairs: int
    relocates: int
    answer: str
    next: str


def _record(ledger: list[str], title: str, summary: str) -> list[str]:
    return ledger + [f"### {title}\n{summary}"]


def _fields(state: AgentState) -> dict[str, Any]:
    """Task state available to the routing decision."""
    return {"issue": state["issue"], "repo_root": state["repo_root"], "ledger": state["ledger"],
            "repairs": state["repairs"], "relocates": state["relocates"], "answer": state["answer"]}


def _phase_node(name: str):
    def node(state: AgentState) -> dict[str, Any]:
        return {"ledger": _record(state["ledger"], name, work(PHASES[name]))}
    return node


def verify_node(state: AgentState) -> dict[str, Any]:
    ledger = _record(state["ledger"], "Verify", work(PHASES["Verify"]))
    repairs, relocates = state["repairs"], state["relocates"]
    # Require an edit and fresh execution evidence before asking the model.
    if repairs >= MAX_REPAIRS or len(tool_log) >= TOOL_BUDGET:
        return {"ledger": ledger, "next": "Finish"}
    repairs += 1
    if not written():
        ledger = _record(ledger, "Verify", "GUARD: no edit has landed; nothing observed proves anything. Back to Edit.")
        return {"ledger": ledger, "repairs": repairs, "next": "Edit"}
    if not ran_since_edit():
        ledger = _record(ledger, "Verify", "GUARD: nothing ran after the last edit. Back to Edit: re-run the reproduction and tests.")
        return {"ledger": ledger, "repairs": repairs, "next": "Edit"}
    walker = {**_fields(state), "ledger": ledger, "repairs": repairs}
    progress = {"progress": "\n\n".join(ledger)}
    # The relocate edge is only put in front of the model once a first repair
    # has also failed, and only once.
    if repairs >= 2 and relocates < MAX_RELOCATES:
        relocates += 1
        walker["relocates"] = relocates
        chosen = select_edge(edges_from("Verify", exclude_kind="forward"), RELOCATE_INTENT, progress, walker, "Verify")
        return {"ledger": ledger, "repairs": repairs, "relocates": relocates,
                "next": chosen.target if chosen else "Edit"}
    chosen = select_edge(edges_from("Verify", exclude_kind="relocate"), REPAIR_INTENT, progress, walker, "Verify")
    return {"ledger": ledger, "repairs": repairs, "next": chosen.target if chosen else "Edit"}


def finish_node(state: AgentState) -> dict[str, Any]:
    repo.cleanup()                         # the reproduction script must not enter the patch
    return {"answer": "\n\n".join(state["ledger"])}


def build_app() -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("Plan", _phase_node("Plan"))
    graph.add_node("Explore", _phase_node("Explore"))
    graph.add_node("Edit", _phase_node("Edit"))
    graph.add_node("Verify", verify_node)
    graph.add_node("Finish", finish_node)
    graph.set_entry_point("Plan")
    for here in ("Plan", "Explore", "Edit"):
        graph.add_edge(here, edges_from(here)[0].target)
    targets = sorted({e.target for e in edges_from("Verify")})
    graph.add_conditional_edges("Verify", lambda s: s["next"], {t: t for t in targets})
    graph.add_edge("Finish", END)
    return graph.compile()


def solve(issue: str, repo_root: str = ".", config: RunnableConfig | None = None) -> str:
    repo.base_dir = repo_root
    history.clear()
    tool_log.clear()
    history.append(SystemMessage(content=SYSTEM.format(issue=issue)))
    out = build_app().invoke(
        {"issue": issue, "repo_root": repo_root, "ledger": [], "repairs": 0,
         "relocates": 0, "answer": "", "next": ""},
        {**(config or {}), "recursion_limit": 100},
    )
    return str(out["answer"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python main.py <issue text> [repo_root]")
    else:
        print(solve(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "."))
