#!/usr/bin/env python3
"""Run the meeting-assistant implementations over the labeled datasets.

For every (implementation, case, repetition) this script prepares an isolated
working directory, copies the case transcript in as meeting_notes.txt, runs
the implementation, and captures its collected mock-tool outputs
(tool_outputs.json) plus wall-clock time and LLM token usage into one
results JSON:

    eval/runs/results_<impl>_<case_id>_r<k>.json

Score the results with score.py. Both implementations call OpenAI gpt-4o, so
OPENAI_API_KEY must be set. Agent pipelines are noisy even at temperature 0 -
use --repeat 3 (or more) and compare means.

Usage:
    python run.py                                  # both impls, all cases
    python run.py --impl byLLM --cases meeting_003 meeting_007 --repeat 3
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent                        # meeting-assistant/
DATASETS_DIR = ROOT / "datasets"
RUNS_DIR = EVAL_DIR / "runs"

# The byLLM implementation is built with the dev-mode jac toolchain.
DEFAULT_JAC_BIN = "/home/xiaoyu/jaseci-gen-sem/jaseci/jac/zig-out/bin"


def implementations():
    jac_bin = os.environ.get("JAC_BIN", DEFAULT_JAC_BIN)
    jac_env = dict(os.environ)
    if Path(jac_bin).is_dir():
        jac_env["PATH"] = f"{jac_bin}:{jac_env.get('PATH', '')}"
    # The CrewAI project uses a src layout; putting it on PYTHONPATH makes
    # `-m meeting_assistant_flow.main` work even when the package is not
    # pip-installed in the interpreter running this script.
    crew_env = dict(os.environ)
    crew_env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "CrewAI" / "src")]
        + ([crew_env["PYTHONPATH"]] if crew_env.get("PYTHONPATH") else [])
    )
    return {
        "CrewAI": {
            "cmd": [sys.executable, "-m", "meeting_assistant_flow.main"],
            "env": crew_env,
        },
        "byLLM": {
            "cmd": ["jac", "run", str(ROOT / "byLLM" / "main.jac")],
            "env": jac_env,
        },
    }


def load_cases(only=None):
    cases = []
    for label_path in sorted(DATASETS_DIR.glob("meeting_*.json")):
        dataset = json.loads(label_path.read_text())
        case_id = dataset["case_id"]
        if only and case_id not in only:
            continue
        transcript = DATASETS_DIR / dataset["transcript_file"]
        if not transcript.is_file():
            print(f"WARNING: skipping {case_id}: missing {transcript.name}")
            continue
        cases.append({"id": case_id, "dataset": label_path, "transcript": transcript})
    return cases


def run_once(impl_name, impl, case, rep, timeout):
    workdir = RUNS_DIR / "work" / impl_name / f"{case['id']}_r{rep}"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    shutil.copy(case["transcript"], workdir / "meeting_notes.txt")

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            impl["cmd"], cwd=workdir, env=impl["env"],
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - t0
        exit_code = -1
        stdout = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = f"TIMEOUT after {timeout}s"

    outputs, output_error = None, None
    outputs_file = workdir / "tool_outputs.json"
    if outputs_file.is_file():
        try:
            outputs = json.loads(outputs_file.read_text())
        except json.JSONDecodeError as exc:
            output_error = f"tool_outputs.json unparseable: {exc}"
    else:
        output_error = "tool_outputs.json not written"

    return {
        "implementation": impl_name,
        "case_id": case["id"],
        "dataset": str(case["dataset"]),
        "repetition": rep,
        "command": impl["cmd"],
        "wall_time_s": round(elapsed, 3),
        "exit_code": exit_code,
        "success": exit_code == 0 and outputs is not None,
        "output_error": output_error,
        "outputs": outputs,
        "token_usage": (outputs or {}).get("token_usage"),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--impl", action="append", choices=["CrewAI", "byLLM"],
                    help="implementation(s) to run (default: both)")
    ap.add_argument("--cases", nargs="+", default=None,
                    help="case ids to run, e.g. meeting_003 (default: all)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="repetitions per (impl, case) - use 3+ for stable means")
    ap.add_argument("--timeout", type=int, default=300,
                    help="per-run timeout in seconds (default: 300)")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set - both implementations call gpt-4o.")

    if (not args.impl or "CrewAI" in args.impl) and not importlib.util.find_spec("crewai"):
        sys.exit(
            f"crewai is not importable with {sys.executable} - run this script "
            "with the Python environment that has CrewAI installed (e.g. "
            "`conda activate jaseci`), or restrict to --impl byLLM."
        )

    impls = implementations()
    selected = {n: impls[n] for n in (args.impl or impls)}
    cases = load_cases(set(args.cases) if args.cases else None)
    if not cases:
        sys.exit("No matching cases found in datasets/.")

    RUNS_DIR.mkdir(exist_ok=True)
    total = len(selected) * len(cases) * args.repeat
    done = 0
    for case in cases:
        for impl_name, impl in selected.items():
            for rep in range(1, args.repeat + 1):
                done += 1
                print(f"[{done}/{total}] {impl_name} / {case['id']} (run {rep}) ... ",
                      end="", flush=True)
                result = run_once(impl_name, impl, case, rep, args.timeout)
                out = RUNS_DIR / f"results_{impl_name}_{case['id']}_r{rep}.json"
                out.write_text(json.dumps(result, indent=2))
                status = "ok" if result["success"] else f"FAILED ({result['output_error'] or result['exit_code']})"
                print(f"{status}  {result['wall_time_s']}s")

    print(f"\nResults written to {RUNS_DIR}/  ->  score with:")
    print(f"  python {EVAL_DIR / 'score.py'} {RUNS_DIR} --judge")


if __name__ == "__main__":
    main()
