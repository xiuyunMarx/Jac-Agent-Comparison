"""The walk: ../Jac/main.jac's `walker CodeAgent`, as a `while` loop.

Same graph, same guards, same routing. The Jac walker visits nodes and its
abilities fire on entry; here `here` is a string and the loop dispatches on it.
The Verify decision is byte for byte the Jac ability's: deterministic guards
first, then the model picks an edge, and the `else` of a failed pick is Edit.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

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


@dataclass
class CodeAgent:
    issue: str
    repo_root: str = "."
    ledger: list[str] = field(default_factory=list)
    repairs: int = 0
    relocates: int = 0
    answer: str = ""

    def record(self, title: str, summary: str) -> None:
        self.ledger.append(f"### {title}\n{summary}")

    def fields(self) -> dict[str, object]:
        """What byLLM's router shows of the walker: its `has` fields, in order."""
        return {"issue": self.issue, "repo_root": self.repo_root, "ledger": self.ledger,
                "repairs": self.repairs, "relocates": self.relocates, "answer": self.answer}

    def progress(self) -> str:
        return "\n\n".join(self.ledger)


def _forward(here: str) -> str:
    """`visit [->:PhaseCapability:->][?:Next]`: the one forward edge."""
    return edges_from(here, exclude_kind=None)[0].target


def route_verify(agent: CodeAgent) -> str:
    """The Verify ability, after `record`: where the walker goes next."""
    # Deterministic guards first, then the model -- the evidence it sees is
    # exactly the evidence that misleads it (old tests all green on an unedited tree).
    if agent.repairs >= MAX_REPAIRS or len(tool_log) >= TOOL_BUDGET:
        return "Finish"
    agent.repairs += 1
    if not written():
        agent.record("Verify", "GUARD: no edit has landed; nothing observed proves anything. Back to Edit.")
        return "Edit"
    if not ran_since_edit():
        agent.record("Verify", "GUARD: nothing ran after the last edit. Back to Edit: re-run the reproduction and tests.")
        return "Edit"
    # The relocate edge is only put in front of the model once a first repair
    # has also failed, and only once.
    if agent.repairs >= 2 and agent.relocates < MAX_RELOCATES:
        agent.relocates += 1
        chosen = select_edge(edges_from("Verify", exclude_kind="forward"), RELOCATE_INTENT,
                             {"progress": agent.progress()}, agent.fields(), "Verify")
        return chosen.target if chosen else "Edit"
    chosen = select_edge(edges_from("Verify", exclude_kind="relocate"), REPAIR_INTENT,
                         {"progress": agent.progress()}, agent.fields(), "Verify")
    return chosen.target if chosen else "Edit"


def walk(agent: CodeAgent) -> None:
    """`root spawn agent`."""
    repo.base_dir = agent.repo_root
    history.clear()
    tool_log.clear()
    history.append({"role": "system", "content": SYSTEM.format(issue=agent.issue)})
    here = "Plan"
    while True:
        if here == "Finish":
            repo.cleanup()                     # the reproduction script must not enter the patch
            agent.answer = agent.progress()
            return
        phase = PHASES[here]
        agent.record(here, work(phase))
        here = route_verify(agent) if here == "Verify" else _forward(here)


def solve(issue: str, repo_root: str = ".") -> str:
    agent = CodeAgent(issue=issue, repo_root=repo_root)
    walk(agent)
    return agent.answer


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python main.py <issue text> [repo_root]")
    else:
        print(solve(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "."))
