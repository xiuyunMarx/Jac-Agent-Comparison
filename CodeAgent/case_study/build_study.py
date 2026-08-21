#!/usr/bin/env python3
"""Turn graded runs into a case study.

    python3 build_study.py --out lite-01 \
        ../swebench_bridge/results/lite-01-byllm \
        ../swebench_bridge/results/lite-01-langgraph

Takes any number of graded run directories and writes three artifacts plus a
README describing them. Grades nothing and calls no model: everything comes from
what the runs already recorded, so regenerating a study is free and repeatable.

The previous version of this study had no generator at all. Its CSV, its JSON,
its per-case metadata and its whole status vocabulary -- TESTS_FAIL,
REGRESSION(P2P), EMPTY_PATCH -- existed only as checked-in files that no code
produced, which meant the numbers could not be re-derived, extended to the other
32 diverging instances, or checked against the runs they claimed to describe.
This file is that missing definition.

**Per-test status is re-derived from the captured logs**, not read from a
grading side-channel. `eval_logs/<id>/test_output.txt` plus the instance's own
log parser is enough to recover FAIL_TO_PASS and PASS_TO_PASS exactly as the
harness saw them, so a study can be built from any graded run whatever graded it,
and no container has to be started to do it.

Outputs, in `--out`:

    verdicts.csv    every instance, one column per framework
    divergence.csv  the subset where the frameworks disagreed -- what you open
    divergence.json the same subset plus full patches and per-test status
    README.md       rates, the agreement partition, and the divergence table
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent / "swebench_bridge"
sys.path.insert(0, str(BRIDGE))

import report as bridge_report  # noqa: E402

# --------------------------------------------------------------------------
# Status vocabulary
#
# One string per (instance, framework). Ordered by what supersedes what: a run
# that never produced a patch cannot also be said to have failed its tests, and
# a patch that would not apply was never measured against them either.
# --------------------------------------------------------------------------

RESOLVED = "RESOLVED"
EMPTY_PATCH = "EMPTY_PATCH"
APPLY_FAIL = "APPLY_FAIL"
REGRESSION = "REGRESSION(P2P)"
SUITE_ERROR = "SUITE_ERROR"
TESTS_FAIL = "TESTS_FAIL"
HARNESS_ERROR = "HARNESS_ERROR"


def classify(entry: dict | None, resolved: bool,
             errored: bool = False, empty: bool = False) -> str:
    """One framework's verdict on one instance.

    `errored` and `empty` come from the harness report rather than from the log,
    and are checked first, because they are the two cases where there may be no
    log at all -- a grade that died before capturing one still has to be
    reported as what it was, not silently downgraded to a test failure.

    The two interesting distinctions, and why each earns its own name:

    `REGRESSION(P2P)` is reserved for a patch that made every FAIL_TO_PASS test
    pass and *then* broke a PASS_TO_PASS one. That is a different animal from
    not fixing the bug: the diagnosis was right and something else came with it,
    which is a scope problem rather than a comprehension one. A patch that
    failed its F2P tests *and* broke P2P ones is just a wrong patch, and calling
    that a regression would flatter it.

    `SUITE_ERROR` is when no PASS_TO_PASS test passed at all. Nothing regressed
    there -- the run collapsed before it could measure anything, usually an
    import error from unparseable source. 0 of 862 matplotlib tests passing is
    not 862 regressions, and averaging it in as one would be worse.
    """
    if errored:
        return HARNESS_ERROR
    if empty:
        return EMPTY_PATCH
    if entry is None:
        # Graded (it is in the report) but no log to re-derive detail from.
        return RESOLVED if resolved else TESTS_FAIL
    if entry.get("error"):
        return HARNESS_ERROR
    if not entry.get("patch_exists"):
        return EMPTY_PATCH
    if not entry.get("patch_successfully_applied"):
        return APPLY_FAIL
    if entry.get("resolved"):
        return RESOLVED
    status = entry.get("tests_status") or {}
    f2p = status.get("FAIL_TO_PASS", {})
    p2p = status.get("PASS_TO_PASS", {})
    p2p_ok, p2p_bad = len(p2p.get("success", [])), len(p2p.get("failure", []))
    if p2p_bad and not p2p_ok:
        return SUITE_ERROR
    if p2p_bad and f2p.get("success") and not f2p.get("failure"):
        return REGRESSION
    return TESTS_FAIL


# Run-level trouble, from what the agent itself recorded. Distinct from the
# verdict: an instance can time out and still leave a patch that resolves.
ERROR_KINDS = (
    ("timed out", "timeout"),
    ("udocker create", "infra(container)"),
    ("udocker pull", "infra(pull)"),
    ("docker", "infra(container)"),
    ("Error code: 400", "api-400"),
    ("BadRequestError", "api-400"),
    ("RateLimit", "api-rate-limit"),
    ("patch extraction", "patch-extraction"),
    ("no result file", "agent-died"),
)


def error_kind(raw: str) -> str:
    for needle, label in ERROR_KINDS:
        if needle in raw:
            return label
    return raw.split(":")[0][:40] if raw else ""


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def load_dataset_rows(dataset: str, split: str) -> dict[str, dict]:
    from datasets import load_dataset

    return {r["instance_id"]: dict(r) for r in load_dataset(dataset, split=split)}


def load_predictions(run_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    path = run_dir / "predictions.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["instance_id"]] = row.get("model_patch") or ""
    return out


def eval_entry(run_dir: Path, iid: str, instance: dict, patch: str) -> dict | None:
    """Re-derive the harness's per-test verdict from the captured log.

    Imported lazily and per instance because `make_test_spec` is the expensive
    part and a study over 300 instances only needs it for the ones it reports.
    """
    log_path = run_dir / "eval_logs" / iid / "test_output.txt"
    if not log_path.exists():
        return None
    if not patch.strip():
        return {"patch_exists": False, "patch_successfully_applied": False,
                "resolved": False}
    from swebench.harness.grading import get_eval_report
    from swebench.harness.utils import make_test_spec

    prediction = {"instance_id": iid, "model_patch": patch,
                  "model_name_or_path": "study"}
    try:
        return get_eval_report(make_test_spec(instance), prediction,
                               str(log_path), True)[iid]
    except Exception as e:  # noqa: BLE001 - a broken log is a datum, not a crash
        return {"patch_exists": True, "patch_successfully_applied": True,
                "resolved": False, "error": f"unparseable log: {type(e).__name__}: {e}"}


DIFF_TARGET = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.MULTILINE)


def files_touched(patch: str) -> list[str]:
    return sorted({m.group(2) for m in DIFF_TARGET.finditer(patch or "")})


# --------------------------------------------------------------------------
# The study
# --------------------------------------------------------------------------


def build(run_dirs: list[Path], dataset: str, split: str) -> dict:
    sides = [bridge_report.load_side(d) for d in run_dirs]
    labels = [s["label"] for s in sides]
    # Prefer the framework each run recorded over its directory name: a run id
    # is a label, but the framework is the thing being compared.
    names = [s["framework"] or s["label"] for s in sides]
    rows = load_dataset_rows(dataset, split)
    patches = [load_predictions(d) for d in run_dirs]

    graded, groups = bridge_report.partition(sides)
    n = len(sides)

    instances: dict[str, dict] = {}
    for iid in sorted(graded):
        instance = rows.get(iid)
        if instance is None:
            continue
        resolvers = tuple(i for i, s in enumerate(sides) if iid in s["resolved"])
        per_side = []
        for i, side in enumerate(sides):
            patch = patches[i].get(iid, "")
            entry = eval_entry(run_dirs[i], iid, instance, patch)
            run = side["runs"].get(iid, {})
            per_side.append({
                "framework": names[i],
                "label": labels[i],
                "status": classify(entry, iid in side["resolved"],
                                   errored=iid in side["error_ids"],
                                   empty=iid in side["empty_ids"]),
                "patch": patch,
                "patch_bytes": len(patch.encode("utf-8")),
                "run_error": error_kind(run.get("error") or ""),
                "steps": run.get("steps"),
                "llm_calls": run.get("llm_calls"),
                "tool_calls": run.get("tool_call_count"),
                "sec": run.get("total_sec"),
                "tests_status": (entry or {}).get("tests_status"),
            })
        distinct = {s["patch"] for s in per_side}
        instances[iid] = {
            "instance_id": iid,
            "repo": instance["repo"],
            "resolved_by": [names[i] for i in resolvers],
            "diverged": 0 < len(resolvers) < n,
            "identical_patches": len(distinct) == 1 and bool(distinct.pop().strip()),
            "gold_patch": instance["patch"],
            "gold_files": files_touched(instance["patch"]),
            "f2p": json.loads(instance["FAIL_TO_PASS"]),
            "p2p": json.loads(instance["PASS_TO_PASS"]),
            "sides": per_side,
        }

    return {
        "dataset": dataset,
        "split": split,
        "frameworks": names,
        "labels": labels,
        "sides": sides,
        "groups": groups,
        "graded": graded,
        "instances": instances,
    }


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def write_verdicts(study: dict, out: Path) -> int:
    names = study["frameworks"]
    header = (["instance_id", "repo", "resolved_by", "diverged"]
              + [f"{n}_status" for n in names]
              + [f"{n}_patch_bytes" for n in names]
              + [f"{n}_run_error" for n in names]
              + ["gold_patch_bytes", "f2p", "p2p"])
    path = out / "verdicts.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in study["instances"].values():
            w.writerow(
                [row["instance_id"], row["repo"], "+".join(row["resolved_by"]),
                 row["diverged"]]
                + [s["status"] for s in row["sides"]]
                + [s["patch_bytes"] for s in row["sides"]]
                + [s["run_error"] for s in row["sides"]]
                + [len(row["gold_patch"].encode("utf-8")),
                   len(row["f2p"]), len(row["p2p"])])
    return len(study["instances"])


def write_divergence(study: dict, out: Path) -> list[dict]:
    names = study["frameworks"]
    diverging = [r for r in study["instances"].values() if r["diverged"]]
    header = (["instance_id", "repo", "resolved_by"]
              + [f"{n}_status" for n in names]
              + [f"{n}_patch_bytes" for n in names]
              + [f"{n}_run_error" for n in names]
              + ["gold_patch_bytes", "gold_files", "identical_patches"])
    with (out / "divergence.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in diverging:
            w.writerow(
                [row["instance_id"], row["repo"], "+".join(row["resolved_by"])]
                + [s["status"] for s in row["sides"]]
                + [s["patch_bytes"] for s in row["sides"]]
                + [s["run_error"] for s in row["sides"]]
                + [len(row["gold_patch"].encode("utf-8")),
                   " ".join(row["gold_files"]), row["identical_patches"]])

    # The JSON carries what the CSV cannot: the patches themselves and the
    # per-test breakdown. That is the difference between "langgraph failed" and
    # "langgraph fixed it and broke five other tests", and only the second is
    # worth anyone's afternoon.
    payload = {
        r["instance_id"]: {
            "repo": r["repo"],
            "resolved_by": r["resolved_by"],
            "identical_patches": r["identical_patches"],
            "gold_patch": r["gold_patch"],
            "gold_files": r["gold_files"],
            "f2p": r["f2p"],
            **{s["framework"]: {k: v for k, v in s.items()
                                if k not in ("framework", "label")}
               for s in r["sides"]},
        }
        for r in diverging
    }
    (out / "divergence.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return diverging


def write_readme(study: dict, out: Path, diverging: list[dict]) -> None:
    sides, names = study["sides"], study["frameworks"]
    n = len(sides)
    graded, groups = study["graded"], study["groups"]
    # The bridge's terminal table labels a side by its run id, which is right
    # there -- you may well be comparing two runs of the same framework. A study
    # is about the implementations, so the group names use those instead, and
    # then match the column headings in the divergence table below.
    named = [{**s, "label": name} for s, name in zip(sides, names)]
    lines: list[str] = []
    add = lines.append

    title = " vs ".join(names)
    add(f"# {title} — SWE-bench {study['dataset'].split('/')[-1]}")
    add("")
    add(f"Generated by `build_study.py` from {n} graded run(s). "
        "Nothing here is hand-maintained: re-run the generator and every number "
        "below is re-derived from the runs.")
    add("")
    add("| run | framework | model | resolved | empty | errored | tokens/instance |")
    add("|---|---|---|---:|---:|---:|---:|")
    for s in sides:
        rate = (f"{len(s['resolved'])}/{s['submitted']}"
                + (f" ({100 * len(s['resolved']) / s['submitted']:.1f}%)"
                   if s["submitted"] else ""))
        add(f"| `{s['label']}` | {s['framework'] or '?'} | {s['model'] or '?'} | "
            f"{rate} | {s['empty']} | {s['errored']} | "
            f"{s['tokens_per_instance']:,.0f} |")
    add("")

    add(f"## Agreement over {len(graded)} instance(s)")
    add("")
    add("Every graded instance falls in exactly one group. Two resolve rates "
        "cannot say whether the implementations solved the *same* problems; "
        "this can.")
    add("")
    add("| resolved by | instances |")
    add("|---|---:|")
    for key in bridge_report.group_order(n):
        add(f"| {bridge_report.group_name(key, named)} | "
            f"{len(groups.get(key, set()))} |")
    add("")

    add(f"## Where they diverged ({len(diverging)} instance(s))")
    add("")
    if not diverging:
        add("No divergence: every graded instance was resolved by all of them "
            "or by none.")
    else:
        add("The instances exactly one subset of the implementations resolved. "
            "Full patches and per-test status for each are in "
            "`divergence.json`.")
        add("")
        add("| instance | resolved by | " + " | ".join(names)
            + " | file the gold patch touches |")
        add("|---|---|" + "---|" * n + "---|")
        for r in diverging:
            gold = ", ".join(f"`{f}`" for f in r["gold_files"][:2]) or "—"
            add(f"| `{r['instance_id']}` | {'+'.join(r['resolved_by'])} | "
                + " | ".join(s["status"] for s in r["sides"])
                + f" | {gold} |")
    add("")

    # An identical patch on both sides that got different verdicts is a grading
    # flake, not a capability difference, and saying so is the difference
    # between a finding and an artifact.
    flakes = [r for r in diverging if r["identical_patches"]]
    if flakes:
        add("### Not capability differences")
        add("")
        add("Byte-identical patches with split verdicts — a grading flake or a "
            "harness error, and no evidence about the implementations:")
        add("")
        for r in flakes:
            add(f"- `{r['instance_id']}`")
        add("")

    errs: dict[str, list[str]] = {}
    for r in study["instances"].values():
        for s in r["sides"]:
            if s["run_error"]:
                errs.setdefault(f"{s['framework']}: {s['run_error']}", []).append(
                    r["instance_id"])
    if errs:
        add("### Run-level trouble")
        add("")
        add("Recorded by the agent itself, and separate from the verdict — an "
            "instance can time out and still leave a patch that resolves.")
        add("")
        add("| | instances |")
        add("|---|---:|")
        for kind, ids in sorted(errs.items(), key=lambda kv: -len(kv[1])):
            add(f"| {kind} | {len(ids)} |")
        add("")

    add("## Files")
    add("")
    add("- `verdicts.csv` — every graded instance, one status column per "
        "implementation.")
    add("- `divergence.csv` — the diverging subset, with patch sizes and the "
        "file the reference fix touches.")
    add("- `divergence.json` — the same subset plus every implementation's full "
        "patch and its FAIL_TO_PASS / PASS_TO_PASS breakdown.")
    add("")
    add("Regenerate with:")
    add("")
    add("```bash")
    add(f"python3 build_study.py --out {out.name} \\")
    add("    " + " \\\n    ".join(
        f"../swebench_bridge/results/{s['label']}" for s in sides))
    add("```")
    add("")

    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build a case study from graded runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("runs", nargs="+", type=Path,
                   help="graded run directories, in presentation order")
    p.add_argument("--out", type=Path, required=True,
                   help="directory to write the study into (created if needed)")
    p.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
    p.add_argument("--split", default="test")
    args = p.parse_args(argv)

    out = (args.out if args.out.is_absolute() else HERE / args.out)
    out.mkdir(parents=True, exist_ok=True)

    study = build([d.resolve() for d in args.runs], args.dataset, args.split)
    total = write_verdicts(study, out)
    diverging = write_divergence(study, out)
    write_readme(study, out, diverging)

    print(f"{total} instance(s), {len(diverging)} diverging -> {out}")
    for name in ("README.md", "verdicts.csv", "divergence.csv", "divergence.json"):
        print(f"  {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
