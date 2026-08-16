#!/usr/bin/env python3
"""Read graded runs and put them next to each other.

    python report.py results/lite-01-byllm
    python report.py results/three-way-01-{byllm,langgraph,openai}

Grades nothing. Every run named here must already have been through grade.py;
this joins each harness report against the `runs.jsonl` the agent itself wrote,
because a resolve rate on its own says nothing about what it cost to get there,
and cost is half the point of comparing implementations.

**Any number of runs.** The comparison is a partition, not a diff: every graded
instance lands in exactly one resolver set -- all of them, some named subset,
one of them, none of them. With two runs that degenerates to the familiar
both/only-A/only-B/neither. With three it is the thing you actually want to
know, because three runs at 40% could be the same twelve instances or
thirty-six different ones and two resolve rates cannot tell those apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def resolve_run_dir(path: Path) -> tuple[Path, Path, str]:
    """Accept a run directory or its predictions.jsonl; return both plus a run id."""
    path = path.resolve()
    predictions = path if path.is_file() else path / "predictions.jsonl"
    if not predictions.exists():
        raise SystemExit(f"no predictions.jsonl at {path}")
    return predictions, predictions.parent / "runs.jsonl", predictions.parent.name


def locate_report(predictions: Path, run_id: str) -> Path | None:
    """The harness-shaped report next to the predictions, whoever wrote it."""
    candidates = sorted(predictions.parent.glob(f"*.{run_id}.json"))
    if not candidates:
        # A run graded under a different --run-id still has exactly one report.
        candidates = [p for p in sorted(predictions.parent.glob("*.json"))
                      if "resolved_ids" in p.read_text(encoding="utf-8")[:4000]]
    return candidates[0] if candidates else None


def load_runs(runs_path: Path) -> dict[str, dict]:
    if not runs_path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["instance_id"]] = row
    return out


def mean(rows: list[dict], key: str) -> float:
    values = [float(r.get(key) or 0) for r in rows]
    return sum(values) / len(values) if values else 0.0


def tokens_of(rows) -> int:
    return sum(int(r.get("prompt_tokens", 0) or 0)
               + int(r.get("completion_tokens", 0) or 0) for r in rows)


def metrics(report_path: Path, runs_path: Path, label: str = "") -> dict:
    """Everything the single-run summary and the comparison table need."""
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
        # The buckets, not just their sizes. An instance the harness could not
        # grade has no log to re-derive anything from, so whoever reports on it
        # has to be able to ask the report which ones those were.
        "empty_ids": set(report.get("empty_patch_ids", [])),
        "error_ids": set(report.get("error_ids", [])),
        "runs": runs,
        "won": won,
        "lost": lost,
        "tokens": tokens,
        "cached_tokens": sum(int(r.get("cached_tokens", 0) or 0)
                             for r in runs.values()),
        "tokens_per_instance": tokens / (len(runs) or 1),
        "tokens_per_resolved": (tokens / len(won)) if won else None,
        "llm_calls": mean(list(runs.values()), "llm_calls"),
        "steps": mean(list(runs.values()), "steps"),
        "tool_calls": mean(list(runs.values()), "tool_call_count"),
        "wall": mean(list(runs.values()), "total_sec"),
        "model": next(iter(runs.values()), {}).get("model", ""),
        "model_name": report.get("model_name_or_path", ""),
        "framework": framework_of(runs, report),
    }


def framework_of(runs: dict[str, dict], report: dict) -> str:
    """Which implementation produced this run.

    Recorded per instance by run_agent.py, but runs graded before that field
    existed only carry it in `model_name_or_path` ("byllm-codeagent/gpt-4o"), so
    fall back to that rather than labelling those runs by their directory name.
    """
    named = next((r.get("framework") for r in runs.values() if r.get("framework")), "")
    if named:
        return named
    model_name = report.get("model_name_or_path", "")
    head = model_name.split("/")[0]
    return head[:-len("-codeagent")] if head.endswith("-codeagent") else ""


def load_side(path: Path, run_id: str = "") -> dict:
    predictions, runs, name = resolve_run_dir(path)
    report = locate_report(predictions, run_id or name)
    if report is None:
        raise SystemExit(
            f"{predictions.parent} has no harness report. Grade it first:\n"
            f"  python {BRIDGE_DIR / 'grade.py'} --predictions {predictions}")
    return metrics(report, runs, label=name)


# --------------------------------------------------------------------------
# The partition
# --------------------------------------------------------------------------


def partition(sides: list[dict]) -> tuple[set[str], dict[tuple[int, ...], set[str]]]:
    """Every graded instance, filed under exactly which sides resolved it."""
    graded: set[str] = set()
    for s in sides:
        graded |= set(s["runs"])
    # A side with no runs.jsonl still has a report; fall back to what it graded
    # so the partition is not silently empty.
    if not graded:
        for s in sides:
            graded |= set(s["resolved"])
    groups: dict[tuple[int, ...], set[str]] = {}
    for iid in graded:
        key = tuple(i for i, s in enumerate(sides) if iid in s["resolved"])
        groups.setdefault(key, set()).add(iid)
    return graded, groups


def group_order(n: int) -> list[tuple[int, ...]]:
    """All resolver sets, widest first: all, then subsets, then singletons, none.

    Enumerated rather than taken from the data so that an empty group still gets
    a line. A zero next to "only openai" is a result; leaving the row out would
    read as "not run".
    """
    keys: list[tuple[int, ...]] = [tuple(range(n))]
    for size in range(n - 1, 0, -1):
        keys += list(combinations(range(n), size))
    keys.append(())
    return keys


def group_name(key: tuple[int, ...], sides: list[dict]) -> str:
    """What to call one resolver set. Two-side wording stays the familiar one."""
    n = len(sides)
    if len(key) == n:
        return "resolved by both" if n == 2 else "resolved by all"
    if not key:
        return "resolved by neither" if n == 2 else "resolved by none"
    if len(key) == 1:
        return f"only {sides[key[0]]['label']}"
    return " + ".join(sides[i]["label"] for i in key)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def summarize(side: dict) -> str:
    resolved, submitted, runs = side["resolved"], side["submitted"], side["runs"]
    lines = [
        "",
        f"SWE-bench report: {side['report_path']}",
        f"  instances submitted : {submitted}",
        f"  resolved            : {len(resolved)}"
        + (f"  ({100 * len(resolved) / submitted:.1f}%)" if submitted else ""),
        f"  unresolved          : {side['unresolved']}",
        f"  empty patches       : {side['empty']}",
        f"  errored             : {side['errored']}",
    ]
    if runs:
        per_resolved = side["tokens_per_resolved"]
        lines += [
            "",
            "Agent cost (from runs.jsonl)",
            f"  total tokens        : {side['tokens']:,}",
            f"  tokens / instance   : {side['tokens_per_instance']:,.0f}",
            "  tokens / resolved   : "
            + (f"{per_resolved:,.0f}" if per_resolved else "n/a"),
            f"  llm calls  (mean)   : {side['llm_calls']:.1f}",
            f"  phases     (mean)   : {side['steps']:.1f}",
            f"  tool calls (mean)   : {side['tool_calls']:.1f}",
            f"  wall clock (mean)   : {side['wall']:.0f}s",
            "",
            f"  resolved   : {mean(side['won'], 'tool_call_count'):.1f} tool calls, "
            f"{mean(side['won'], 'steps'):.1f} phases",
            f"  unresolved : {mean(side['lost'], 'tool_call_count'):.1f} tool calls, "
            f"{mean(side['lost'], 'steps'):.1f} phases",
        ]
        errors = Counter((r.get("error") or "").split(":")[0]
                         for r in runs.values() if r.get("error"))
        if errors:
            lines += ["", "Agent-side errors"]
            lines += [f"  {n:>4}  {kind}" for kind, n in errors.most_common(8)]
    return "\n".join(lines)


def compare_table(sides: list[dict], show: int = 12) -> str:
    """Side-by-side comparison of two or more graded runs."""
    if len(sides) < 2:
        raise ValueError("a comparison needs at least two runs")
    n = len(sides)
    col = max(14, *(len(s["label"]) for s in sides))
    # 2 leading spaces + a 22-wide label column, then one col-wide value per
    # side joined by two spaces.
    rule = "=" * (24 + n * col + 2 * (n - 1))

    def row(label: str, *values: str) -> str:
        return f"  {label:<22}" + "  ".join(f"{v:>{col}}" for v in values)

    def cells(fmt) -> list[str]:
        return [fmt(s) for s in sides]

    def pct(m: dict) -> str:
        total = m["submitted"]
        return f"{len(m['resolved'])}" + (
            f" ({100 * len(m['resolved']) / total:.1f}%)" if total else "")

    def toks(m: dict) -> str:
        v = m["tokens_per_resolved"]
        return f"{v:,.0f}" if v else "n/a"

    graded, groups = partition(sides)

    lines = [
        "",
        rule,
        f"  {('A/B' if n == 2 else f'{n}-way'):<22}"
        + "  ".join(f"{s['label']:>{col}}" for s in sides),
        rule,
        row("model", *cells(lambda s: s["model"] or "?")),
        row("submitted", *cells(lambda s: str(s["submitted"]))),
        row("resolved", *cells(pct)),
        row("unresolved", *cells(lambda s: str(s["unresolved"]))),
        row("empty patches", *cells(lambda s: str(s["empty"]))),
        row("errored", *cells(lambda s: str(s["errored"]))),
        "",
        row("total tokens", *cells(lambda s: f"{s['tokens']:,}")),
        row("tokens / instance", *cells(lambda s: f"{s['tokens_per_instance']:,.0f}")),
        row("tokens / resolved", *cells(toks)),
        row("llm calls (mean)", *cells(lambda s: f"{s['llm_calls']:.1f}")),
        row("phases (mean)", *cells(lambda s: f"{s['steps']:.1f}")),
        row("tool calls (mean)", *cells(lambda s: f"{s['tool_calls']:.1f}")),
        row("wall clock (mean)", *cells(lambda s: f"{s['wall']:.0f}s")),
        "",
        f"  Agreement over {len(graded)} instance(s) run by "
        + ("both" if n == 2 else f"all {n}"),
    ]

    keys = group_order(n)
    names = {k: group_name(k, sides) for k in keys}
    width = max(24, *(len(v) for v in names.values()))
    lines += [f"    {names[k]:<{width}} : {len(groups.get(k, set()))}"
              for k in keys]

    def listing(title: str, ids: set[str]) -> list[str]:
        if not ids:
            return []
        shown = sorted(ids)[:show]
        out = ["", f"    {title}:"] + [f"      {i}" for i in shown]
        if len(ids) > show:
            out.append(f"      ... and {len(ids) - show} more")
        return out

    # The disagreement set only: full agreement and total failure are counts,
    # not lists anyone opens.
    for key in keys:
        if 0 < len(key) < n:
            lines += listing(names[key], groups.get(key, set()))

    if not any(s["runs"] for s in sides):
        lines += ["", "  (no runs.jsonl on any side; only the harness "
                      "totals above are comparable)"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Summarize one graded run, or compare any number of them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("runs", nargs="+", type=Path,
                   help="graded run directories (or their predictions.jsonl)")
    p.add_argument("--run-id", default="",
                   help="report id, if it differs from the directory name")
    p.add_argument("--show", type=int, default=12,
                   help="instance ids to list per disagreement group")
    args = p.parse_args(argv)

    sides = [load_side(path, args.run_id) for path in args.runs]
    if len(sides) == 1:
        print(summarize(sides[0]))
    else:
        print(compare_table(sides, show=args.show))
    return 0


if __name__ == "__main__":
    sys.exit(main())
