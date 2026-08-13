#!/usr/bin/env python3
"""Grade a predictions file with the official SWE-bench harness, then report.

Two things happen here and they are deliberately separate:

  * `swebench.harness.run_evaluation` decides what resolved. It is the official
    harness, run unmodified, in fresh containers built from the instance images
    -- nothing the agent did to its workspace during inference is visible to it.
  * the report joins that verdict against runs.jsonl, which is what the agent
    itself recorded: phases, tool calls, tokens. A resolve rate on its own says
    nothing about what it cost to get there, and cost is the point of an A/B
    between two agent frameworks.

`--compare RUN_A RUN_B` skips straight to that A/B, joining two already-graded
runs. It grades nothing itself, so the two sides must already have been through
the harness -- which is what compare.py does for you.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parent
DEFAULT_SWEBENCH = BRIDGE_DIR.parent / "SWE-bench"


def harness_installed() -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", "import swebench"],
        capture_output=True, text=True,
    )
    return probe.returncode == 0


def grade_without_daemon(args: argparse.Namespace) -> int:
    """The same evaluation, driven through udocker instead of the Docker API."""
    import grade_local

    return grade_local.main([
        "--predictions", str(args.predictions),
        "--dataset", args.dataset,
        "--split", args.split,
        "--run-id", args.run_id,
        "--runtime", "udocker",
        "--udocker", args.udocker,
        "--udocker-dir", str(args.udocker_dir),
        "--workers", str(args.max_workers),
        "--timeout", str(args.timeout),
        *(["--instance-ids", *args.instance_ids] if args.instance_ids else []),
    ])


def run_harness(args: argparse.Namespace) -> int:
    argv = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", args.dataset,
        "--split", args.split,
        "--predictions_path", str(args.predictions),
        "--run_id", args.run_id,
        "--max_workers", str(args.max_workers),
        "--timeout", str(args.timeout),
    ]
    if args.instance_ids:
        argv += ["--instance_ids", *args.instance_ids]
    print("$ " + " ".join(argv), flush=True)
    # cwd, not --report_dir: the harness creates that directory and then writes
    # the report relative to the process's working directory anyway. Running
    # from the prediction's own directory is what actually puts the report
    # beside the run it belongs to.
    return subprocess.run(argv, cwd=str(args.predictions.parent)).returncode


# --------------------------------------------------------------------------
# Reading a graded run back
# --------------------------------------------------------------------------


def locate_report(predictions: Path, run_id: str, swebench: Path) -> Path | None:
    """The harness names the report <model_name_or_path>.<run_id>.json."""
    model_names = {
        json.loads(line)["model_name_or_path"]
        for line in predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    roots = [predictions.parent, Path.cwd(), swebench]
    for name in sorted(model_names):
        stem = name.replace("/", "__")
        for root in roots:
            candidate = root / f"{stem}.{run_id}.json"
            if candidate.exists():
                return candidate
    for root in roots:
        matches = sorted(root.glob(f"*.{run_id}.json"))
        if matches:
            return matches[0]
    return None


def load_runs(runs_path: Path) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    if runs_path.exists():
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                runs[row["instance_id"]] = row
    return runs


def mean(rows: list[dict], key: str) -> float:
    vals = [float(r.get(key, 0) or 0) for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def tokens_of(rows) -> int:
    return sum(int(r.get("prompt_tokens", 0) or 0)
               + int(r.get("completion_tokens", 0) or 0) for r in rows)


def metrics(report_path: Path, runs_path: Path, label: str = "") -> dict:
    """Everything both the single-run summary and the A/B table need."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    resolved = set(report.get("resolved_ids", []))
    runs = load_runs(runs_path)
    won = [r for iid, r in runs.items() if iid in resolved]
    lost = [r for iid, r in runs.items() if iid not in resolved]
    submitted = report.get("submitted_instances", 0)
    tokens = tokens_of(runs.values())
    return {
        "label": label or report_path.parent.name,
        "report_path": report_path,
        "resolved": resolved,
        "submitted": submitted,
        "unresolved": report.get("unresolved_instances", 0),
        "empty": report.get("empty_patch_instances", 0),
        "errored": report.get("error_instances", 0),
        "runs": runs,
        "won": won,
        "lost": lost,
        "tokens": tokens,
        "tokens_per_instance": tokens / (len(runs) or 1),
        "tokens_per_resolved": (tokens / len(won)) if won else None,
        "llm_calls": mean(list(runs.values()), "llm_calls"),
        "steps": mean(list(runs.values()), "steps"),
        "tool_calls": mean(list(runs.values()), "tool_call_count"),
        "wall": mean(list(runs.values()), "total_sec"),
        "model": next(iter(runs.values()), {}).get("model", ""),
    }


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def summarize(report_path: Path, runs_path: Path) -> str:
    m = metrics(report_path, runs_path)
    resolved, submitted, runs = m["resolved"], m["submitted"], m["runs"]

    lines = [
        "",
        f"SWE-bench report: {report_path}",
        f"  instances submitted : {submitted}",
        f"  resolved            : {len(resolved)}"
        + (f"  ({100 * len(resolved) / submitted:.1f}%)" if submitted else ""),
        f"  unresolved          : {m['unresolved']}",
        f"  empty patches       : {m['empty']}",
        f"  errored             : {m['errored']}",
    ]
    if runs:
        per_resolved = m["tokens_per_resolved"]
        lines += [
            "",
            "Agent cost (from runs.jsonl)",
            f"  total tokens        : {m['tokens']:,}",
            f"  tokens / instance   : {m['tokens_per_instance']:,.0f}",
            "  tokens / resolved   : "
            + (f"{per_resolved:,.0f}" if per_resolved else "n/a"),
            f"  llm calls  (mean)   : {m['llm_calls']:.1f}",
            f"  phases     (mean)   : {m['steps']:.1f}",
            f"  tool calls (mean)   : {m['tool_calls']:.1f}",
            f"  wall clock (mean)   : {m['wall']:.0f}s",
            "",
            f"  resolved   : {mean(m['won'], 'tool_call_count'):.1f} tool calls, "
            f"{mean(m['won'], 'steps'):.1f} phases",
            f"  unresolved : {mean(m['lost'], 'tool_call_count'):.1f} tool calls, "
            f"{mean(m['lost'], 'steps'):.1f} phases",
        ]
        errors = Counter(
            (r.get("error") or "").split(":")[0]
            for r in runs.values() if r.get("error")
        )
        if errors:
            lines += ["", "Agent-side errors"]
            lines += [f"  {n:>4}  {kind}" for kind, n in errors.most_common(8)]
    return "\n".join(lines)


def compare_table(a: dict, b: dict, show: int = 12) -> str:
    """Side-by-side A/B of two graded runs.

    The paired columns are the point: two resolve rates alone do not say whether
    the frameworks solved the *same* instances, and the disagreement set is what
    anyone reading an A/B actually goes and opens.
    """
    wa, wb = len(a["label"]), len(b["label"])
    col = max(wa, wb, 14)

    def row(label: str, va: str, vb: str) -> str:
        return f"  {label:<22}{va:>{col}}  {vb:>{col}}"

    def pct(m: dict) -> str:
        n = m["submitted"]
        return f"{len(m['resolved'])}" + (
            f" ({100 * len(m['resolved']) / n:.1f}%)" if n else ""
        )

    def toks(m: dict) -> str:
        v = m["tokens_per_resolved"]
        return f"{v:,.0f}" if v else "n/a"

    graded = set(a["runs"]) | set(b["runs"])
    ra, rb = a["resolved"], b["resolved"]
    both = ra & rb
    only_a = ra - rb
    only_b = rb - ra
    neither = graded - ra - rb

    lines = [
        "",
        "=" * (26 + 2 * col),
        f"  {'A/B':<22}{a['label']:>{col}}  {b['label']:>{col}}",
        "=" * (26 + 2 * col),
        row("model", a["model"] or "?", b["model"] or "?"),
        row("submitted", str(a["submitted"]), str(b["submitted"])),
        row("resolved", pct(a), pct(b)),
        row("unresolved", str(a["unresolved"]), str(b["unresolved"])),
        row("empty patches", str(a["empty"]), str(b["empty"])),
        row("errored", str(a["errored"]), str(b["errored"])),
        "",
        row("total tokens", f"{a['tokens']:,}", f"{b['tokens']:,}"),
        row("tokens / instance", f"{a['tokens_per_instance']:,.0f}",
            f"{b['tokens_per_instance']:,.0f}"),
        row("tokens / resolved", toks(a), toks(b)),
        row("llm calls (mean)", f"{a['llm_calls']:.1f}", f"{b['llm_calls']:.1f}"),
        row("phases (mean)", f"{a['steps']:.1f}", f"{b['steps']:.1f}"),
        row("tool calls (mean)", f"{a['tool_calls']:.1f}", f"{b['tool_calls']:.1f}"),
        row("wall clock (mean)", f"{a['wall']:.0f}s", f"{b['wall']:.0f}s"),
        "",
        f"  Agreement over {len(graded)} instance(s) run by both",
    ]
    names = [f"only {a['label']}", f"only {b['label']}"]
    width = max(24, *(len(n) for n in names))
    lines += [
        f"    {'resolved by both':<{width}} : {len(both)}",
        f"    {names[0]:<{width}} : {len(only_a)}",
        f"    {names[1]:<{width}} : {len(only_b)}",
        f"    {'resolved by neither':<{width}} : {len(neither)}",
    ]

    def listing(title: str, ids: set[str]) -> list[str]:
        if not ids:
            return []
        shown = sorted(ids)[:show]
        out = ["", f"    {title}:"] + [f"      {i}" for i in shown]
        if len(ids) > show:
            out.append(f"      ... and {len(ids) - show} more")
        return out

    lines += listing(names[0], only_a)
    lines += listing(names[1], only_b)

    if not graded:
        lines += ["", "  (no runs.jsonl on either side; only the harness "
                      "totals above are comparable)"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def resolve_run_dir(path: Path) -> tuple[Path, Path, str]:
    """Accept a run directory or its predictions.jsonl; return both plus a run id."""
    path = path.resolve()
    predictions = path if path.is_file() else path / "predictions.jsonl"
    if not predictions.exists():
        raise SystemExit(f"no predictions.jsonl at {path}")
    return predictions, predictions.parent / "runs.jsonl", predictions.parent.name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grade coding-agent predictions with the official harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--predictions", type=Path,
                   help="predictions.jsonl written by run_agent.py")
    p.add_argument("--compare", type=Path, nargs=2, metavar=("RUN_A", "RUN_B"),
                   help="two already-graded run directories to put side by side; "
                        "grades nothing itself")
    p.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
    p.add_argument("--split", default="test")
    p.add_argument("--run-id", default="",
                   help="evaluation run id (default: the inference run's directory name)")
    p.add_argument("--instance-ids", nargs="*", default=[])
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=1800,
                   help="per-instance test timeout in seconds")
    p.add_argument("--runtime", default="docker", choices=["docker", "udocker"],
                   help="'docker' runs the official swebench harness, which "
                        "needs a reachable daemon; 'udocker' runs the same "
                        "evaluation through grade_local.py, which needs neither "
                        "a daemon nor root")
    p.add_argument("--udocker", default="udocker")
    p.add_argument("--udocker-dir", type=Path, default=Path.home() / ".udocker")
    p.add_argument("--swebench", type=Path, default=DEFAULT_SWEBENCH)
    p.add_argument("--report-only", action="store_true",
                   help="skip evaluation and re-print the summary for an existing report")
    args = p.parse_args(argv)
    if not args.predictions and not args.compare:
        p.error("one of --predictions or --compare is required")
    args.swebench = args.swebench.resolve()
    if args.predictions:
        args.predictions = args.predictions.resolve()
        if not args.run_id:
            args.run_id = args.predictions.parent.name
        args.runs = args.predictions.parent / "runs.jsonl"
    return args


def compare_only(args: argparse.Namespace) -> int:
    sides = []
    for path in args.compare:
        predictions, runs, run_id = resolve_run_dir(path)
        report = locate_report(predictions, args.run_id or run_id, args.swebench)
        if report is None:
            raise SystemExit(
                f"{predictions.parent} has no harness report "
                f"(*.{args.run_id or run_id}.json). Grade it first with:\n"
                f"  python {Path(__file__).name} --predictions {predictions}"
            )
        sides.append(metrics(report, runs, label=run_id))
    print(compare_table(*sides))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compare:
        return compare_only(args)

    if not args.predictions.exists():
        raise SystemExit(f"no predictions at {args.predictions}")
    if not harness_installed():
        raise SystemExit(
            "the swebench package is not importable. Install it with:\n"
            f"  pip install -e {args.swebench}"
        )

    if not args.report_only:
        code = grade_without_daemon(args) if args.runtime == "udocker" else run_harness(args)
        if code != 0:
            print(f"\nthe harness exited {code}; "
                  "any report it did write is summarised below", file=sys.stderr)

    report = locate_report(args.predictions, args.run_id, args.swebench)
    if report is None:
        raise SystemExit(
            f"no report matching *.{args.run_id}.json under {args.swebench}"
        )
    print(summarize(report, args.runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
