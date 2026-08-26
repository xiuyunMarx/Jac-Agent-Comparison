# SPDX-License-Identifier: Apache-2.0
"""Wiring: one agent, one run, and the record the benchmark reads.

Everything interesting is in agent.py -- this is the seam the SWE-bench bridge
imports. ../swebench_bridge/swe_entry.py resolves the agent from
$CODEAGENT_HOME and imports `solve`, `active_model_name` and `DEFAULT_MODEL`
from a module named `orchestrator`, so those names live here with byLLM's
`RunResult` field set behind them.

The other three sides have an orchestrator worth the name: a phase graph, a
router, a walk, and a per-phase ReAct loop, 460 to 880 lines of it. On this side
the runtime owns the loop and the agent owns the workflow, so what is left over
is a dataclass, an `asyncio.run`, and a CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Sequence

from agent import DEFAULT_MAX_STEPS, CodeAgent, Report
from llm import (
    DEFAULT_MODEL,
    active_model_name,
    get_client,
    set_client,
    set_model,
    token_usage,
)
from tools.common import ToolCall, get_tool_calls, reset_tool_log

__all__ = [
    "DEFAULT_MODEL",
    "CodeAgent",
    "Report",
    "RunResult",
    "active_model_name",
    "arun",
    "main",
    "set_client",
    "set_model",
    "solve",
]


@dataclass
class RunResult:
    """What one agent run produced.

    Typed rather than a dict, with the same fields as byLLM's, LangGraph's and
    openai_sdk's RunResult, so the eval harness reads one shape on every side.
    `cached_tokens` is populated here: it comes off the usage block of every
    response, which the meter in telemetry.py sees.

    `steps` counts generated-method calls -- `resolve` plus its repairs -- which
    is what `--max-steps` bounds on this side. The other three count phase
    visits. Neither number is the other, and the reports say which is which.
    """

    objective: str
    answer: str = ""
    steps: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    resolved: bool = False


async def arun(
    objective: str, repo_root: str = "", max_steps: int = DEFAULT_MAX_STEPS
) -> RunResult:
    """One run, on this event loop."""
    reset_tool_log()
    token_usage.reset()
    root = os.path.realpath(repo_root or os.getcwd())
    agent = CodeAgent(
        repo_root=root,
        llm=get_client(),
        context={
            "workspace": (
                f"The repository is at {root}. Every path you pass to a tool is "
                "relative to that directory."
            )
        },
    )
    report: Report | None = None
    error = ""
    try:
        report = await agent.run(objective, max_steps)
    except Exception as e:  # noqa: BLE001 - the patch is taken from the workspace either way
        error = f"{type(e).__name__}: {e}"
        sys.stderr.write(f"[run] {error}\n")

    if report is not None:
        answer = report.rendered()
    else:
        # A run that died still owes the log an account of itself. The plan is
        # the one piece of state the model maintained that survives, and the
        # tool log below says what it actually did.
        answer = f"(the run failed: {error})\n\n## Plan\n{agent.plan.show_plan()}"

    totals = token_usage.totals()
    return RunResult(
        objective=objective,
        answer=answer,
        steps=agent.steps,
        tool_calls=get_tool_calls(),
        llm_calls=totals["llm_calls"],
        prompt_tokens=totals["prompt_tokens"],
        completion_tokens=totals["completion_tokens"],
        cached_tokens=totals["cached_tokens"],
        resolved=bool(report and report.resolved),
    )


def solve(
    objective: str, repo_root: str = "", max_steps: int = DEFAULT_MAX_STEPS
) -> RunResult:
    """The shim's entry point: synchronous, one run per process.

    `asyncio.run` rather than a shared loop because that is exactly the lifetime
    -- swe_entry.py runs one instance per process and exits.
    """
    return asyncio.run(arun(objective, repo_root, max_steps))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NOOA coding agent.")
    parser.add_argument("--task", required=True, help="The task to be solved.")
    parser.add_argument(
        "--repo-path",
        default="",
        help="Path to the repository to work in. Defaults to the current directory.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="How many generated-method calls the run may spend.",
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    outcome = solve(args.task, args.repo_path, args.max_steps)
    # stdout is the answer channel -- the eval harness reads it -- so the
    # telemetry line goes to stderr.
    print(outcome.answer)
    sys.stderr.write(
        f"[{outcome.steps} steps, {len(outcome.tool_calls)} tool calls, "
        f"{outcome.llm_calls} llm calls, "
        f"{outcome.prompt_tokens}+{outcome.completion_tokens} tokens]\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
