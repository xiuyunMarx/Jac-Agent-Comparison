#!/usr/bin/env python3
"""Run every email agent over every mock mailbox, then score them together.

The sibling comparisons each have one of these; this project had only a shell
loop in the README, which is why its third arm went a year with a single batch
run against it. Same shape as `meeting-assistant/eval/run.py`: a table of
implementations, one subprocess per (impl, batch), results left where
`score.py` already looks for them.

Three things this does that the README loop does not:

  * **One model on every side.** $OPENAI_MODEL_NAME is exported to all three
    arms. Left unset, crewai falls back to gpt-4o-mini and the other two to
    gpt-4o -- a ~17x per-token price difference that reads as a framework
    result. See eval/README.md, "Token cost".
  * **The CrewAI arm gets its own interpreter.** It pins langgraph 1.x, which
    cannot coexist with the 0.3.x the other benchmarks need, so it lives in
    `CrewAI-LangGraph/.venv`. That venv is used when present.
  * **Every arm is scored in one invocation**, because score.py rewrites
    summary.json from only the runs it was given.

Usage:

    python run.py                                   # 3 impls x 6 batches, then score
    python run.py --impl openai_sdk --batches batch_001
    python run.py --no-score                        # run only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent                      # Email-Auto-response/
DATASETS_DIR = ROOT / "mock_mailbox" / "datasets"


def crewai_python() -> str:
    """The CrewAI arm's own interpreter, or this one if it has no venv.

    Not a preference: `CrewAI-LangGraph/requirements.txt` pins
    langgraph>=1.2.5, and CodeAgent/langgraph plus YT-Navigator pin 0.3.5. One
    environment cannot hold both, so this arm keeps its own.
    """
    venv = ROOT / "CrewAI-LangGraph" / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


def implementations() -> dict[str, dict]:
    """Each arm: where it runs, how it is launched, how the dataset reaches it.

    byLLM's entry point is `nodes.jac`, not the `main.jac` its jac.toml
    declares -- that file is a hello-world stub. It also takes no argument; the
    dataset travels in $EMAIL_DATASET, which the other two accept as a fallback
    so one variable drives all three.
    """
    return {
        "byLLM": {
            "dir": ROOT / "byLLM",
            "cmd": ["jac", "run", "nodes.jac"],
            "takes_dataset_arg": False,
        },
        "CrewAI-LangGraph": {
            "dir": ROOT / "CrewAI-LangGraph",
            "cmd": [crewai_python(), "main.py"],
            "takes_dataset_arg": True,
        },
        "openai_sdk": {
            "dir": ROOT / "openai_sdk",
            "cmd": [sys.executable, "main.py"],
            "takes_dataset_arg": True,
        },
    }


def load_batches(only: set[str] | None) -> list[Path]:
    batches = []
    for path in sorted(DATASETS_DIR.glob("batch_*.json")):
        if only and path.stem not in only:
            continue
        batches.append(path)
    return batches


def run_once(impl_name: str, impl: dict, dataset: Path, timeout: int) -> dict:
    env = dict(os.environ)
    env["EMAIL_DATASET"] = str(dataset)
    cmd = list(impl["cmd"]) + ([str(dataset)] if impl["takes_dataset_arg"] else [])

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=impl["dir"], env=env, timeout=timeout,
                              capture_output=True, text=True)
        elapsed = time.perf_counter() - t0
        exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - t0
        exit_code = -1
        stdout = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = f"TIMEOUT after {timeout}s"
    except FileNotFoundError as exc:
        return {"implementation": impl_name, "case_id": dataset.stem, "success": False,
                "wall_time_s": 0.0, "exit_code": 127, "error": str(exc),
                "results_path": None, "stdout_tail": "", "stderr_tail": str(exc)}

    # Each arm writes mock_output/results_<case_id>.json itself; case_id comes
    # from inside the dataset, so read it rather than assuming the filename.
    case_id = json.loads(dataset.read_text()).get("case_id", dataset.stem)
    results_path = impl["dir"] / "mock_output" / f"results_{case_id}.json"

    return {
        "implementation": impl_name,
        "case_id": case_id,
        "dataset": str(dataset),
        "command": cmd,
        "wall_time_s": round(elapsed, 3),
        "exit_code": exit_code,
        # A crash after the inbox is written still leaves usable results: both
        # byLLM and openai_sdk save, then re-raise. So success is "the file is
        # there", not "the exit code was 0" -- and the code is reported anyway.
        "success": results_path.is_file(),
        "results_path": str(results_path) if results_path.is_file() else None,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", action="append",
                    choices=["byLLM", "CrewAI-LangGraph", "openai_sdk"],
                    help="implementation(s) to run (default: all three)")
    ap.add_argument("--batches", nargs="+", default=None,
                    help="batch stems to run, e.g. batch_001 (default: all)")
    ap.add_argument("--timeout", type=int, default=1200,
                    help="per-run timeout in seconds (default: 1200)")
    ap.add_argument("--no-score", action="store_true", help="skip the scoring step")
    ap.add_argument("--judge", action="store_true",
                    help="add LLM-judged draft quality to the scoring step")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set - every arm calls a model. "
                 "Set a dummy value when pointing OPENAI_BASE_URL at a local server.")
    if not os.environ.get("OPENAI_MODEL_NAME"):
        print("WARNING: OPENAI_MODEL_NAME is unset. crewai will fall back to "
              "gpt-4o-mini and the other two to gpt-4o, and the comparison will "
              "measure the model rather than the framework.", file=sys.stderr)

    impls = implementations()
    selected = {n: impls[n] for n in (args.impl or impls)}
    batches = load_batches(set(args.batches) if args.batches else None)
    if not batches:
        sys.exit(f"No matching batches in {DATASETS_DIR}.")

    total, done, failures = len(selected) * len(batches), 0, 0
    records = []
    for dataset in batches:
        for impl_name, impl in selected.items():
            done += 1
            print(f"[{done}/{total}] {impl_name} / {dataset.stem} ... ", end="", flush=True)
            record = run_once(impl_name, impl, dataset, args.timeout)
            records.append(record)
            if record["success"]:
                print(f"ok ({record['wall_time_s']}s)")
            else:
                failures += 1
                print(f"FAILED (exit {record['exit_code']})")
                tail = (record.get("stderr_tail") or "").strip().splitlines()
                for line in tail[-4:]:
                    print(f"    {line}")

    print(f"\n{total - failures}/{total} runs produced results.")

    if args.no_score:
        return 1 if failures else 0

    # One invocation over every arm: score.py rebuilds summary.json from just
    # the runs it is handed, so scoring them separately would leave a summary
    # describing whichever arm went last.
    out_dirs = [str(impl["dir"] / "mock_output") for impl in selected.values()
                if (impl["dir"] / "mock_output").is_dir()]
    if not out_dirs:
        sys.exit("No results to score.")
    cmd = [sys.executable, str(EVAL_DIR / "score.py")] + out_dirs
    if args.judge:
        cmd.append("--judge")
    print(f"\nScoring: {' '.join(Path(d).parent.name for d in out_dirs)}\n")
    return subprocess.run(cmd).returncode or (1 if failures else 0)


if __name__ == "__main__":
    raise SystemExit(main())
