#!/usr/bin/env python3
"""Run the YT-Navigator agent implementations over the shared question set.

Both implementations answer against the same Postgres/PGVector database and
the same questions file, and both emit result records in the shared schema
(YT-Navigator/benchmark/schemas.py), so their outputs are scored side by side
with the shared evaluator.

    python run.py                          # both implementations, then score
    python run.py --impl byllm             # one implementation only
    python run.py --judge                  # add LLM-as-judge answer scoring
    python run.py --langgraph-python /path/to/yt-navigator-venv/bin/python

Prerequisites (see README.md): the database loaded with a snapshot, a
questions file, OPENAI_API_KEY, and per-implementation dependencies. Database
credentials are read from YT-Navigator/.env (override with --env-file).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent                      # YTNavigator/
LANGGRAPH_DIR = ROOT / "YT-Navigator"
BYLLM_DIR = ROOT / "byLLM"
EVALUATE = LANGGRAPH_DIR / "benchmark" / "evaluate.py"


def parse_env_file(path):
    """Parse KEY=VALUE lines from a .env file (no shell interpolation)."""
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def run_impl(name, cmd, cwd, env, timeout):
    """Run one implementation, streaming its output; return (ok, wall_seconds)."""
    print(f"\n=== {name}: {' '.join(str(c) for c in cmd)} (cwd={cwd}) ===")
    started = time.perf_counter()
    try:
        proc = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, timeout=timeout)
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"ERROR: {name} timed out after {timeout}s")
        ok = False
    except FileNotFoundError as e:
        print(f"ERROR: {name}: {e}")
        ok = False
    elapsed = round(time.perf_counter() - started, 1)
    print(f"=== {name}: {'ok' if ok else 'FAILED'} in {elapsed}s ===")
    return ok, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl", choices=["both", "byllm", "langgraph"], default="both")
    parser.add_argument(
        "--questions",
        default=str(ROOT / "datasets" / "questions.jsonl"),
        help="Shared questions JSONL (default: datasets/questions.jsonl)",
    )
    parser.add_argument("--channel", default="", help="Channel id (default: the only channel in the database)")
    parser.add_argument("--out-dir", default=str(EVAL_DIR / "out"), help="Directory for result files")
    parser.add_argument(
        "--env-file",
        default=str(LANGGRAPH_DIR / ".env"),
        help="Env file with POSTGRES_* / OPENAI_API_KEY (default: YT-Navigator/.env)",
    )
    parser.add_argument(
        "--langgraph-python",
        default=sys.executable,
        help="Python interpreter with YT-Navigator's dependencies installed (default: this one)",
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Per-implementation timeout in seconds")
    parser.add_argument("--no-score", action="store_true", help="Skip the scoring step")
    parser.add_argument("--judge", action="store_true", help="Add LLM-as-judge scoring (needs reference answers)")
    args = parser.parse_args()

    questions = Path(args.questions)
    if not questions.is_file():
        example = LANGGRAPH_DIR / "benchmark" / "questions.example.jsonl"
        sys.exit(
            f"Questions file not found: {questions}\n"
            f"Author one for your snapshotted channel first (start from {example})."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update({k: v for k, v in parse_env_file(Path(args.env_file)).items() if k not in os.environ})

    results = []
    if args.impl in ("both", "byllm"):
        out = out_dir / "results_byllm.jsonl"
        byllm_env = dict(env)
        byllm_env.update(
            {
                "YTNAV_QUESTIONS": str(questions.resolve()),
                "YTNAV_OUTPUT": str(out.resolve()),
                "YTNAV_CHANNEL": args.channel,
            }
        )
        ok, _ = run_impl("byLLM", ["jac", "run", "main.jac"], BYLLM_DIR, byllm_env, args.timeout)
        if ok:
            results.append(out)

    if args.impl in ("both", "langgraph"):
        out = out_dir / "results_langgraph.jsonl"
        cmd = [
            args.langgraph_python, "manage.py", "benchmark_run",
            str(questions.resolve()), "-o", str(out.resolve()), "--framework", "langgraph",
        ]
        if args.channel:
            cmd += ["--channel", args.channel]
        ok, _ = run_impl("LangGraph", cmd, LANGGRAPH_DIR, env, args.timeout)
        if ok:
            results.append(out)

    if not results:
        sys.exit("No implementation produced results; nothing to score.")

    if args.no_score:
        print(f"\nResults: {', '.join(str(r) for r in results)}")
        print(f"Score later with: python {EVAL_DIR / 'score.py'} "
              f"{' '.join(str(r) for r in results)} --questions {questions}")
        return

    score_cmd = [sys.executable, str(EVALUATE)] + [str(r) for r in results] + [
        "--questions", str(questions),
        "--report", str(out_dir / "report.json"),
    ]
    if args.judge:
        score_cmd.append("--judge")
    print()
    subprocess.run(score_cmd, env=env, check=False)


if __name__ == "__main__":
    main()
