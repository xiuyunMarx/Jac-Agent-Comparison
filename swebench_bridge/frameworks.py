#!/usr/bin/env python3
"""Which agent implementations exist, and how each one is launched.

Three implementations of *one* agent -- same five phases, same ten tools, same
prompts, same string contracts. What differs is how the phase graph, the ReAct
loop and the accounting are expressed:

    byllm      ../byLLM       Jac + byLLM         a walker over a phase graph
    langgraph  ../langgraph   Python + LangGraph  a compiled StateGraph
    openai     ../openai_sdk  Python, no framework  a while loop

The registry is the whole of the fork. Everything else in the bridge -- the
workspace, the container, the objective text, the preparation step, the patch
extraction, the grading -- is shared, because a difference in the score has to
be a difference between the agents rather than between drivers that drifted.

Adding a fourth implementation is one entry here. Nothing in the bridge counts
frameworks or assumes there are two of them; that assumption is what the
previous version of this directory was built on and it is why comparing three
meant rewriting it.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parent
ROOT = BRIDGE_DIR.parent


@dataclass(frozen=True)
class Framework:
    """Everything that differs between the implementations, which is very little.

    `marker` is the file that must exist under the agent home for it to be that
    agent at all -- checked before a run rather than discovered at instance 200.
    `runner` names which interpreter option launches `entry`.
    """

    name: str
    home: Path
    entry: Path
    marker: str
    runner: str  # "jac" | "python"
    blurb: str

    def argv(self, runner_bin: str, *extra: str) -> list[str]:
        """How this framework's shim is spawned. The only fork in the driver."""
        if self.runner == "jac":
            return [runner_bin, "run", str(self.entry), *extra]
        return [runner_bin, str(self.entry), *extra]

    def check(self, home: Path, runner_bin: str) -> str:
        """Why this framework cannot run here, or "" if it can."""
        if not self.entry.exists():
            return f"missing shim: {self.entry}"
        if not (home / self.marker).exists():
            return f"no {self.name} agent at {home} (expected {self.marker})"
        if shutil.which(runner_bin) is None:
            return f"the {self.runner} runner '{runner_bin}' is not on PATH"
        return ""


# The two Python implementations share one shim. It resolves the agent from
# $CODEAGENT_HOME, which the driver sets per framework, and both projects export
# solve / active_model_name / DEFAULT_MODEL from a module named `orchestrator`.
# The Jac one needs its own because `jac` is a self-contained binary carrying its
# own Python, so this interpreter cannot import orchestrator.jac at all.
PY_SHIM = BRIDGE_DIR / "swe_entry.py"
JAC_SHIM = BRIDGE_DIR / "swe_entry.jac"

FRAMEWORKS: dict[str, Framework] = {
    "byllm": Framework(
        name="byllm",
        home=ROOT / "byLLM",
        entry=JAC_SHIM,
        marker="orchestrator.jac",
        runner="jac",
        blurb="Jac + byLLM: a walker over a phase graph",
    ),
    "langgraph": Framework(
        name="langgraph",
        home=ROOT / "langgraph",
        entry=PY_SHIM,
        marker="orchestrator.py",
        runner="python",
        blurb="Python + LangGraph: a compiled StateGraph",
    ),
    "openai": Framework(
        name="openai",
        home=ROOT / "openai_sdk",
        entry=PY_SHIM,
        marker="orchestrator.py",
        runner="python",
        blurb="Python, no framework: a while loop over the OpenAI SDK",
    ),
}

# The order comparisons are presented in: the Jac original, the framework port,
# then the no-framework baseline the other two are measured against.
ORDER = ["byllm", "langgraph", "openai"]

NAMES = sorted(FRAMEWORKS)


def get(name: str) -> Framework:
    if name not in FRAMEWORKS:
        raise SystemExit(
            f"unknown framework {name!r}; known: {', '.join(NAMES)}")
    return FRAMEWORKS[name]


def ordered(names: list[str]) -> list[str]:
    """Sort a selection into ORDER, keeping anything unknown at the end."""
    return sorted(names, key=lambda n: (ORDER.index(n) if n in ORDER else 99, n))
