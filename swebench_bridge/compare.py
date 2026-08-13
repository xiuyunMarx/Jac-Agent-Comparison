#!/usr/bin/env python3
"""Run both coding agents over the same SWE-bench instances, grade both, compare.

    python compare.py --repo psf/requests --limit 2 --run-id smoke

which is the same as doing this by hand:

    python run_agent.py --framework byllm     --run-id smoke-byllm     --instance-ids ...
    python run_agent.py --framework langgraph --run-id smoke-langgraph --instance-ids ...
    python evaluate.py  --predictions results/smoke-byllm/predictions.jsonl
    python evaluate.py  --predictions results/smoke-langgraph/predictions.jsonl
    python evaluate.py  --compare results/smoke-byllm results/smoke-langgraph

Two things this does that running it by hand does not:

**The instance set is resolved once, up front, and pinned.** Both sides are
handed the same explicit `--instance-ids`, so `--limit 20` cannot mean a
different twenty on the second side because the dataset revision moved or a
filter behaved differently. An A/B over two different instance sets is not an
A/B.

**The frameworks run one after another, never at once.** They would otherwise
contend for the same cores, the same disk and the same docker daemon, and the
wall-clock and token numbers are meant to be compared against each other.

Everything else is delegated: inference is run_agent.py's job and grading is
evaluate.py's, both called in-process so their validation and their exit codes
are the real ones. Unrecognised flags are forwarded to run_agent.py, so
`--workers 8 --max-steps 12` and friends work here too.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import evaluate
import run_agent

BRIDGE_DIR = Path(__file__).resolve().parent
ORDER = ["byllm", "langgraph"]


def resolve_instances(args: argparse.Namespace, passthrough: list[str]) -> list[dict]:
    """Apply the selection flags once, through run_agent's own semantics."""
    probe = run_agent.parse_args([
        *passthrough,
        "--framework", args.frameworks[0],
        "--run-id", "instance-probe",
        "--dataset", args.dataset,
        "--split", args.split,
        *(["--limit", str(args.limit)] if args.limit else []),
        *(["--repo", *args.repo] if args.repo else []),
        *(["--instance-ids", *args.instance_ids] if args.instance_ids else []),
    ])
    return run_agent.load_instances(probe)


def side_run_id(run_id: str, framework: str) -> str:
    return f"{run_id}-{framework}"


def infer(args: argparse.Namespace, framework: str, ids: list[str],
          passthrough: list[str]) -> int:
    return run_agent.main([
        *passthrough,
        "--runtime", args.runtime,
        "--framework", framework,
        "--run-id", side_run_id(args.run_id, framework),
        "--output-dir", str(args.output_dir),
        "--dataset", args.dataset,
        "--split", args.split,
        "--instance-ids", *ids,
    ])


def grade(args: argparse.Namespace, framework: str) -> int:
    run_dir = args.output_dir / side_run_id(args.run_id, framework)
    return evaluate.main([
        "--predictions", str(run_dir / "predictions.jsonl"),
        "--runtime", args.runtime,
        "--dataset", args.dataset,
        "--split", args.split,
        "--max-workers", str(args.eval_workers),
        "--timeout", str(args.eval_timeout),
    ])


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n== {text}\n{'=' * 78}", flush=True)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Run both agents over the same instances and compare them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Any other flag is forwarded to run_agent.py unchanged.",
        # Off because this parser forwards what it does not recognise: with
        # abbreviation on, a run_agent.py flag that merely prefixes one of ours
        # (`--framework` against `--frameworks`) would be silently swallowed
        # here instead of reaching run_agent.py or tripping the guard below.
        allow_abbrev=False,
    )
    p.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"),
                   help="each side runs as <run-id>-<framework>")
    p.add_argument("--frameworks", nargs="+", default=ORDER, choices=ORDER,
                   help="which agents to run, in this order")
    p.add_argument("--output-dir", type=Path, default=BRIDGE_DIR / "results")
    p.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
    p.add_argument("--split", default="test")
    p.add_argument("--instance-ids", nargs="*", default=[],
                   help="run only these instances")
    p.add_argument("--repo", nargs="*", default=[],
                   help="run only instances from these repos, e.g. psf/requests")
    p.add_argument("--limit", type=int, default=0,
                   help="run only the first N instances after filtering")
    p.add_argument("--skip-eval", action="store_true",
                   help="inference only; grade later with evaluate.py")
    p.add_argument("--runtime", default="docker", choices=["docker", "udocker"],
                   help="container runtime for both inference and grading; "
                        "'udocker' needs no daemon, no root and no docker group")
    p.add_argument("--eval-workers", type=int, default=4,
                   help="--max-workers for the grading harness")
    p.add_argument("--eval-timeout", type=int, default=1800,
                   help="per-instance test timeout during grading")
    args, passthrough = p.parse_known_args(argv)
    args.output_dir = args.output_dir.resolve()
    # These belong to both sides identically, so compare.py owns them and they
    # must not also arrive through the passthrough.
    for flag in ("--run-id", "--output-dir", "--framework", "--dataset",
                 "--split", "--instance-ids", "--repo", "--limit", "--runtime"):
        if flag in passthrough:
            p.error(f"{flag} is set by compare.py; pass it to compare.py itself")
    return args, passthrough


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_args(argv)

    instances = resolve_instances(args, passthrough)
    if not instances:
        raise SystemExit("no instances matched the selection")
    ids = [i["instance_id"] for i in instances]
    print(f"A/B over {len(ids)} instance(s): {', '.join(ids[:6])}"
          + (f" ... (+{len(ids) - 6})" if len(ids) > 6 else ""))

    for framework in args.frameworks:
        banner(f"inference: {framework}")
        code = infer(args, framework, ids, passthrough)
        if code != 0:
            # Sequential on purpose, so a broken first side would otherwise
            # burn the full cost of the second before anyone saw it.
            print(f"\n{framework} inference exited {code}; stopping here",
                  file=sys.stderr)
            return code

    if args.skip_eval:
        print("\n--skip-eval: grade these when ready with")
        for framework in args.frameworks:
            run_dir = args.output_dir / side_run_id(args.run_id, framework)
            print(f"  python evaluate.py --predictions {run_dir / 'predictions.jsonl'}")
        return 0

    for framework in args.frameworks:
        banner(f"grading: {framework}")
        code = grade(args, framework)
        if code != 0:
            print(f"\n{framework} grading exited {code}", file=sys.stderr)

    if len(args.frameworks) == 2:
        banner("A/B")
        return evaluate.main([
            "--compare",
            *(str(args.output_dir / side_run_id(args.run_id, f))
              for f in args.frameworks),
        ])
    return 0


if __name__ == "__main__":
    sys.exit(main())
