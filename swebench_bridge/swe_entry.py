#!/usr/bin/env python3
"""One SWE-bench instance, run through the LangGraph coding agent.

The Python twin of swe_entry.jac, and deliberately the same shape: job file in,
result file out, one instance per process. The byLLM side has no choice about
that -- `jac` carries its own interpreter and cannot be imported -- but this
side keeps it for two reasons of its own:

  * the agent holds its repository binding and its token counter in module
    globals (`orchestrator.rt`, `orchestrator.token_usage`), so N instances in
    N threads of one process would interleave into each other's state;
  * a crash costs one instance instead of the whole run, and the workspace it
    already edited is still on disk for the driver to take a patch from.

stdout is a transcript, not a channel: LangChain and httpx both log to it.
"""

from __future__ import annotations

import json
import os
import sys
import time

# Resolved before the agent is imported: a bare module name is looked up against
# this file's own directory first, and the agent lives in a different project.
AGENT_HOME = os.path.realpath(
    os.environ.get("CODEAGENT_HOME")
    or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "langgraph")
)
if AGENT_HOME not in sys.path:
    sys.path.insert(0, AGENT_HOME)

from orchestrator import DEFAULT_MODEL, active_model_name, solve  # noqa: E402


def configured_model() -> str:
    """The model this run will call, without requiring it to be constructible.

    byLLM builds its `Model` at import and `active_model_name` just reads the
    attribute; this side defers construction, so asking too early would raise on
    a missing API key and lose the error report that the caller needs. Falling
    back to the configured name reports the same string for the same run.
    """
    try:
        return active_model_name()
    except Exception:
        return os.environ.get("CODEAGENT_MODEL", DEFAULT_MODEL)


def run_job(job_path: str) -> dict:
    """Read the job, run the agent, and return what the driver needs to record."""
    with open(job_path, "r", encoding="utf-8") as jf:
        job = json.load(jf)
    objective = str(job.get("objective", ""))
    repo_root = str(job.get("repo_root", ""))
    max_steps = int(job.get("max_steps", 10) or 10)
    record: dict = {
        "instance_id": str(job.get("instance_id", "")),
        "model": configured_model(),
        "answer": "",
        "steps": 0,
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tool_calls": [],
        "tool_call_count": 0,
        "error": "",
    }
    began = time.time()
    try:
        outcome = solve(objective, repo_root, max_steps)
        calls = [{"name": c.name, "args": c.args} for c in outcome.tool_calls]
        record["answer"] = outcome.answer
        record["steps"] = outcome.steps
        record["llm_calls"] = outcome.llm_calls
        record["prompt_tokens"] = outcome.prompt_tokens
        record["completion_tokens"] = outcome.completion_tokens
        record["tool_calls"] = calls
        record["tool_call_count"] = len(calls)
    except Exception as e:  # noqa: BLE001 - one instance must not end the run
        # The patch is taken from the workspace either way, so a crashed run
        # still yields whatever edits it had already made.
        record["error"] = f"{type(e).__name__}: {e}"
    record["wall_clock_sec"] = round(time.time() - began, 3)
    return record


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python swe_entry.py <job.json> <result.json>")
        return 2
    result = run_job(argv[0])
    with open(argv[1], "w", encoding="utf-8") as rf:
        rf.write(json.dumps(result, indent=2))
    status = result["error"] or "ok"
    print(f"[swe_entry] {result['instance_id']} {status} "
          f"steps={result['steps']} tools={result['tool_call_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
