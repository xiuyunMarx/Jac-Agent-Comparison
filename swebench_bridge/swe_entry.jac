"""One SWE-bench instance, run through the byLLM coding agent.

The agent cannot be imported from an ordinary Python: the `jac` CLI is a
self-contained binary carrying its own interpreter, and `jaclang.byllm` exists
only inside it. So the driver crosses the boundary the other way -- it writes a
job file, runs this module under `jac run`, and reads the result file back.

Everything interesting is a file rather than stdout because byLLM and litellm
both log to the console, so stdout is not a channel anything can parse.

The Jac twin of swe_entry.py, which serves both Python implementations. The two
write the same result keys, so runs.jsonl has one shape on every side and the
reports are directly comparable.
"""

import json;
import os;
import sys;
import time;

# Top-level statements run in source order, so this glob lands before the import
# below it resolves. It has to: a bare module name is looked up against this
# file's own directory first, and the agent lives in a different project.
glob AGENT_HOME: str = os.path.realpath(
    os.environ.get("CODEAGENT_HOME")
    or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "byLLM")
);
glob _PATH_READY: bool = (
    (AGENT_HOME in sys.path) or (sys.path.insert(0, AGENT_HOME) is None)
);

import from orchestrator { solve, active_model_name }


"""Read the job, run the agent, and return what the driver needs to record."""
def run_job(job_path: str) -> dict[str, object] {
    with open(job_path, "r", encoding="utf-8") as jf {
        job = json.loads(jf.read());
    }
    objective = str(job.get("objective", ""));
    repo_root = str(job.get("repo_root", ""));
    max_steps = int(job.get("max_steps", 10) or 10);
    record: dict[str, object] = {
        "instance_id": str(job.get("instance_id", "")),
        "model": active_model_name(),
        "answer": "",
        "steps": 0,
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "tool_calls": [],
        "tool_call_count": 0,
        "error": ""
    };
    began = time.time();
    try {
        outcome = solve(objective, repo_root, max_steps);
        calls: list[dict[str, object]] = [
            {"name": c.name, "args": c.args} for c in outcome.tool_calls
        ];
        record["answer"] = outcome.answer;
        record["steps"] = outcome.steps;
        record["llm_calls"] = outcome.llm_calls;
        record["prompt_tokens"] = outcome.prompt_tokens;
        record["completion_tokens"] = outcome.completion_tokens;
        # Part of prompt_tokens, not extra to it. Recorded so the cost table can
        # say what a run actually paid for rather than what it sent.
        record["cached_tokens"] = int(getattr(outcome, "cached_tokens", 0) or 0);
        record["tool_calls"] = calls;
        record["tool_call_count"] = len(calls);
    } except Exception as e {
        # One instance blowing up must not cost the other 299. The patch is
        # taken from the workspace either way, so a crashed run still yields
        # whatever edits it had already made.
        record["error"] = f"{type(e).__name__}: {str(e)}";
    }
    record["wall_clock_sec"] = round(time.time() - began, 3);
    return record;
}


with entry:__main__ {
    argv = sys.argv[1:];
    if len(argv) != 2 {
        print("usage: jac run swe_entry.jac <job.json> <result.json>");
        sys.exit(2);
    }
    result = run_job(argv[0]);
    with open(argv[1], "w", encoding="utf-8") as rf {
        rf.write(json.dumps(result, indent=2));
    }
    status = result["error"] or "ok";
    print(f"[swe_entry] {result['instance_id']} {status} "
          f"steps={result['steps']} tools={result['tool_call_count']}");
}
