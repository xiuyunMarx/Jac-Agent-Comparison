"""The walk: ../Jac/main.jac's `walker CodeAgent`, as a compiled StateGraph.

Same graph, same guards, same routing. The Jac walker's `has` fields are the
graph state; each phase ability is a node; Plan -> Explore -> Edit -> Verify are
fixed edges, and Verify's decision -- deterministic guards first, then the
model picks an edge, `else` Edit -- is made inside the Verify node and read
back by its conditional edge, because a LangGraph router must not mutate state
and the Jac ability mutates `repairs`, `relocates` and the ledger on the way.
"""

from __future__ import annotations

import sys
from typing import Any, TypedDict

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph

from nodes import PHASES, TOOL_BUDGET, edges_from, history, ran_since_edit, repo, tool_log, written
from phase_agent import select_edge, work

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


class AgentState(TypedDict):
    """`walker CodeAgent`'s `has` fields, plus where Verify decided to go."""

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
    """What byLLM's router shows of the walker: its `has` fields, in order."""
    return {"issue": state["issue"], "repo_root": state["repo_root"], "ledger": state["ledger"],
            "repairs": state["repairs"], "relocates": state["relocates"], "answer": state["answer"]}


def _phase_node(name: str):
    def node(state: AgentState) -> dict[str, Any]:
        return {"ledger": _record(state["ledger"], name, work(PHASES[name]))}
    return node


def verify_node(state: AgentState) -> dict[str, Any]:
    ledger = _record(state["ledger"], "Verify", work(PHASES["Verify"]))
    repairs, relocates = state["repairs"], state["relocates"]
    # Deterministic guards first, then the model -- the evidence it sees is
    # exactly the evidence that misleads it (old tests all green on an unedited tree).
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
    # `visit [->:PhaseCapability:->][?:Next]`: the declared forward edges.
    for here in ("Plan", "Explore", "Edit"):
        graph.add_edge(here, edges_from(here)[0].target)
    targets = sorted({e.target for e in edges_from("Verify")})
    graph.add_conditional_edges("Verify", lambda s: s["next"], {t: t for t in targets})
    graph.add_edge("Finish", END)
    return graph.compile()


def solve(issue: str, repo_root: str = ".") -> str:
    repo.base_dir = repo_root
    history.clear()
    tool_log.clear()
    history.append(SystemMessage(content=SYSTEM.format(issue=issue)))
    out = build_app().invoke(
        {"issue": issue, "repo_root": repo_root, "ledger": [], "repairs": 0,
         "relocates": 0, "answer": "", "next": ""},
        {"recursion_limit": 100},
    )
    return str(out["answer"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python main.py <issue text> [repo_root]")
    else:
        print(solve(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "."))
