"""Run every arm on every BFCL task, one process at a time; one JSON result per (arm, task).

    cd ReAct_loop
    set -a; . ../.env; set +a                      # OLLAMA_API_KEY
    python eval/run.py                              # all arms, the first 50 tasks
    python eval/run.py --arms jac nooa --limit 5    # a subset
    python eval/score.py                            # -> eval/out/report.md

A run whose result JSON already exists without an error is skipped; --force reruns it.
An arm exits with RATE_LIMIT_EXIT when the search backend keeps answering HTTP 429: that run
is recorded as an error and the sweep stops, so rerunning the same command resumes it.
Token usage comes from each arm's own "Usage:" line, except the Jac arm, whose
calls byLLM writes to the file named by BYLLM_USAGE_LOG.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "out"
ENVS = Path.home() / "miniconda3" / "envs"
TIMEOUT_S = 3600   # a run can stall ~31 min on one rate-limited search, plus 20 s between live searches
N_TASKS = 50   # verification set: the first 50 of the 100 BFCL web_search tasks
RATE_LIMIT_EXIT = 75   # every arm's exit status when its search is still rate-limited after retries

ARMS: dict[str, tuple[Path, list[str]]] = {
    "jac": (ROOT / "Jac", [str(ENVS / "jac-main/bin/jac"), "run", "main.jac"]),
    "openai": (ROOT, [str(ENVS / "jaseci/bin/python"), "OpenaiSDK/main.py"]),
    "langgraph": (ROOT, [str(ENVS / "jaseci/bin/python"), "LangGraph/main.py"]),
    "nooa": (ROOT, [str(ENVS / "nooa/bin/python"), "NOOA/main.py"]),
}


def task_ids() -> list[str]:
    lines = (ROOT / "data" / "BFCL_v4_web_search.json").read_text().splitlines()
    return [json.loads(line)["id"] for line in lines if line.strip()]


def field(pattern: str, text: str) -> str | None:
    found = re.search(pattern, text, re.M | re.S)
    return found.group(1).strip() if found else None


def usage_from_log(path: Path) -> dict[str, int]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    return {
        "calls": len(rows),
        "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in rows),
        "completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in rows),
    }


def run_one(arm: str, task: str) -> dict:
    cwd, cmd = ARMS[arm]
    tag = f"{arm}_{task}"
    log = OUT / f"{tag}.usage.jsonl"
    log.unlink(missing_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if arm == "jac":
        env["BYLLM_USAGE_LOG"] = str(log)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd + [task], cwd=cwd, env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
        code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        code, stdout, stderr = -9, e.stdout or "", f"timeout after {TIMEOUT_S}s"
        stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
    wall = round(time.monotonic() - t0, 1)
    (OUT / f"{tag}.stdout.txt").write_text(stdout + (f"\n--- stderr ---\n{stderr}" if stderr else ""))

    verdict = field(r"^Verdict: (\w+)", stdout)
    # an endpoint failure is not a task failure: record it as an error so the run is retried
    provider_error = field(r"^- solve failed: ([^\n]*(?:AuthenticationError|RateLimitError|APIConnectionError|APITimeoutError|"
                           r"InternalServerError|ServiceUnavailable|Error code: (?:401|403|429|5\d\d))[^\n]*)$", stdout)
    usage_line = field(r"^Usage: (.*?)$", stdout)
    if arm == "jac":
        usage = usage_from_log(log)
    else:
        usage = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", usage_line or "")}
    return {
        "arm": arm,
        "task": task,
        "verdict": verdict,
        "answer": field(r"^Answer: (.*?)^Gold: ", stdout),
        "gold": field(r"^Gold: (.*?)$", stdout),
        "tool_calls": int(field(r"^Tool calls: (\d+)", stdout) or 0),
        "diagnostics": field(r"^Diagnostics:\n(.*)", stdout),
        "calls": usage.get("calls", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "wall_s": wall,
        "exit": code,
        "error": (provider_error or "")[:300] or (
            None if code == 0 and verdict else (stderr.strip().splitlines() or ["no verdict printed"])[-1][:300]
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--tasks", nargs="+", help="task ids (default: the first N_TASKS)")
    ap.add_argument("--limit", type=int, help="first N tasks only")
    ap.add_argument("--force", action="store_true", help="rerun tasks that already have a result")
    args = ap.parse_args()
    if not os.environ.get("OLLAMA_API_KEY"):
        sys.exit("OLLAMA_API_KEY is not set (set -a; . ../.env; set +a)")

    OUT.mkdir(parents=True, exist_ok=True)
    tasks = (args.tasks or task_ids()[:N_TASKS])[: args.limit]
    jobs = []
    # task-major: every arm runs a task before the next task starts, so the arms query the web
    # at about the same time and a partial run stays balanced across arms
    for task in tasks:
        for arm in args.arms:
            prior = OUT / f"{arm}_{task}.json"
            if args.force or not prior.exists() or json.loads(prior.read_text())["error"]:
                jobs.append((arm, task))
    print(f"{len(jobs)} runs ({len(tasks)} tasks x {args.arms}), one at a time", flush=True)
    for i, (arm, task) in enumerate(jobs, 1):
        r = run_one(arm, task)
        (OUT / f"{arm}_{task}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        print(f"[{i}/{len(jobs)}] {arm:9} {task:14} {r['verdict'] or 'ERROR':8} calls={r['calls']} "
              f"tok={r['prompt_tokens']}+{r['completion_tokens']} tools={r['tool_calls']} {r['wall_s']}s"
              + (f"  error: {r['error']}" if r["error"] else ""), flush=True)
        if r["exit"] == RATE_LIMIT_EXIT:
            sys.exit("search backend is rate-limiting (HTTP 429); stopped. Rerun the same command later to resume.")


if __name__ == "__main__":
    main()
