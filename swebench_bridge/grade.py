#!/usr/bin/env python3
"""Grade predictions with the official harness's verdict, without its daemon.

    python grade.py --predictions results/smoke-openai/predictions.jsonl

`swebench.harness.run_evaluation` talks to the Docker API directly, so on a
machine where you are not in the `docker` group it cannot run at all. This is
the same evaluation driven through runtime.py instead, which means it also works
under udocker -- pure userspace, no daemon, no root.

What is *not* reimplemented is the part that decides the answer. The test
specification, the log parsers and the resolution rule all come from the
installed swebench package:

    make_test_spec(instance)   -> the eval script and the F2P/P2P sets
    get_eval_report(...)       -> parses the log and decides `resolved`

so a verdict here is the verdict the official harness would give for the same
captured log. What this file owns is only how the container is obtained and how
the log is produced:

  1. a fresh container from the instance image -- never the inference
     workspace, so nothing the agent did to its own tree can reach the score;
  2. `model_patch` applied to /testbed, trying the same command ladder the
     official harness tries;
  3. the instance's own eval_script run inside, which resets the graded test
     files, applies the test patch and runs the suite;
  4. the captured output written in the layout get_logs_eval expects.

Two things it does that the previous grader split across two files. Every
verdict is appended to `eval_results.jsonl` as it lands, so an interrupted grade
costs one instance rather than the run and re-running skips what is already
decided. And each image is retired the moment its instance is graded, with a
disk floor in front of every unpack -- grading a 300-instance run means ~500 GB
of images if none are ever dropped, which is how the previous full run died
two-thirds of the way through with `No space left on device`.

The report lands next to the predictions with the harness's own filename,
`<model_name_or_path>.<run_id>.json`, in the harness's own shape -- so report.py
reads it without knowing which grader produced it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from runtime import (
    RUNTIMES,
    DiskGate,
    RuntimeConfig,
    StepError,
    build_runtime,
    free_gb,
    log,
)

BRIDGE_DIR = Path(__file__).resolve().parent

# The ladder the official harness walks, in its order. A patch that needs --3way
# or --reject still counts as applied there, so it must here too.
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --3way",
    "git apply --verbose --reject",
    "patch --batch --forward --fuzz=5 -p1 -i",
]

CONTAINER_PATCH = "/testbed/model_patch.diff"
CONTAINER_EVAL = "/testbed/run_eval.sh"

_append_lock = threading.Lock()


def swebench_bits():
    """Imported lazily so --help works without the harness installed."""
    from swebench.harness.constants import APPLY_PATCH_FAIL, APPLY_PATCH_PASS
    from swebench.harness.grading import get_eval_report
    from swebench.harness.utils import make_test_spec

    return make_test_spec, get_eval_report, APPLY_PATCH_PASS, APPLY_PATCH_FAIL


def harness_installed() -> bool:
    try:
        swebench_bits()
        return True
    except Exception:  # noqa: BLE001
        return False


def load_predictions(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["instance_id"]] = row
    return rows


def load_done(path: Path) -> dict[str, dict]:
    """Verdicts already decided, so an interrupted grade resumes."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["instance_id"]] = row["entry"]
    return out


def record_verdict(path: Path, iid: str, entry: dict) -> None:
    with _append_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"instance_id": iid, "entry": entry}) + "\n")


# --------------------------------------------------------------------------
# One instance
# --------------------------------------------------------------------------


def grade_one(args, runtime, instance: dict, prediction: dict) -> dict:
    """Run one instance's evaluation and return its report_map entry."""
    make_test_spec, get_eval_report, APPLY_PASS, APPLY_FAIL = swebench_bits()
    iid = instance["instance_id"]
    spec = make_test_spec(instance)
    log_dir = args.log_dir / iid
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "test_output.txt"
    patch = prediction.get("model_patch") or ""

    if not patch.strip():
        # Nothing to apply and nothing to run; the harness records this as an
        # empty patch rather than as an error.
        log_path.write_text("(empty patch; nothing was run)\n", encoding="utf-8")
        return {iid: {"patch_is_None": False, "patch_exists": False,
                      "patch_successfully_applied": False, "resolved": False}}

    container = f"grade-{iid}-{args.run_id}"[:120]
    ws = args.workspaces / iid
    transcript: list[str] = []
    try:
        args.disk.wait(iid)
        runtime.ensure_image(spec.image)
        ws = runtime.create(container, spec.image, ws)
        (ws / Path(CONTAINER_PATCH).name).write_text(patch, encoding="utf-8")
        (ws / Path(CONTAINER_EVAL).name).write_text(spec.eval_script, encoding="utf-8")

        applied = False
        for cmd in GIT_APPLY_CMDS:
            code, out = runtime.exec_script(
                container, f"cd /testbed && {cmd} {CONTAINER_PATCH}", args.timeout)
            transcript.append(f"$ {cmd} {CONTAINER_PATCH}\n{out}")
            if code == 0:
                transcript.append(f"{APPLY_PASS}:\napplied with `{cmd}`")
                applied = True
                break
        if not applied:
            transcript.append(f"{APPLY_FAIL}:\nnone of the apply commands succeeded")
            log_path.write_text("\n".join(transcript), encoding="utf-8")
            return {iid: {"patch_is_None": False, "patch_exists": True,
                          "patch_successfully_applied": False, "resolved": False}}

        # The eval script carries its own START/END markers, which is exactly
        # what get_logs_eval slices the parseable region out of.
        code, out = runtime.exec_script(
            container, f"chmod +x {CONTAINER_EVAL} && bash {CONTAINER_EVAL}",
            args.timeout)
        transcript.append(out)
        log_path.write_text("\n".join(transcript), encoding="utf-8")
        return get_eval_report(spec, prediction, str(log_path), True)
    except StepError as e:
        transcript.append(f">>>>> Tests Errored\n{e}")
        log_path.write_text("\n".join(transcript), encoding="utf-8")
        return {iid: {"patch_is_None": False, "patch_exists": True,
                      "patch_successfully_applied": False, "resolved": False,
                      "error": str(e)}}
    finally:
        try:
            runtime.destroy(container, ws, keep_workspace=False)
        except Exception:  # noqa: BLE001 - teardown must not mask a verdict
            pass
        if args.retire_images:
            # The moment this instance is decided its image is dead weight, and
            # the next instance needs the room more than the cache is worth.
            try:
                runtime.remove_image(spec.image)
            except Exception as e:  # noqa: BLE001
                log(f"  (removing {spec.image} failed: {type(e).__name__}: {e})")


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def build_report(predictions: dict, results: dict[str, dict],
                 model_name: str) -> dict:
    """The harness's own report shape, so report.py cannot tell the difference."""
    resolved, unresolved, empty, errored, completed = [], [], [], [], []
    for iid, entry in sorted(results.items()):
        row = entry.get(iid, {})
        if not row.get("patch_exists"):
            empty.append(iid)
        elif row.get("error"):
            errored.append(iid)
        elif row.get("resolved"):
            resolved.append(iid)
            completed.append(iid)
        else:
            unresolved.append(iid)
            completed.append(iid)
    return {
        "total_instances": len(predictions),
        "submitted_instances": len(predictions),
        "completed_instances": len(completed),
        "resolved_instances": len(resolved),
        "unresolved_instances": len(unresolved),
        "empty_patch_instances": len(empty),
        "error_instances": len(errored),
        "completed_ids": completed,
        "resolved_ids": resolved,
        "unresolved_ids": unresolved,
        "empty_patch_ids": empty,
        "error_ids": errored,
        "model_name_or_path": model_name,
        "graded_by": "swebench_bridge/grade.py",
    }


def report_path_for(predictions: Path, model_name: str, run_id: str) -> Path:
    return predictions.parent / f"{model_name.replace('/', '__')}.{run_id}.json"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grade predictions with the official verdict, no daemon needed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
    p.add_argument("--split", default="test")
    p.add_argument("--run-id", default="",
                   help="default: the prediction directory's name")
    p.add_argument("--runtime", default="udocker", choices=sorted(RUNTIMES))
    p.add_argument("--udocker", default="udocker")
    p.add_argument("--udocker-dir", type=Path, default=Path.home() / ".udocker")
    p.add_argument("--workers", type=int, default=2,
                   help="grades in flight; each holds an unpacked container")
    p.add_argument("--timeout", type=float, default=1800,
                   help="per-instance ceiling for the test run")
    p.add_argument("--pull-timeout", type=float, default=3600)
    p.add_argument("--copy-timeout", type=float, default=1800)
    p.add_argument("--network", default="")
    p.add_argument("--instance-ids", nargs="*", default=[])
    p.add_argument("--min-free-gb", type=float, default=6.0,
                   help="a worker waits until the disk has this much room "
                        "before unpacking another container")
    p.add_argument("--keep-images", action="store_true",
                   help="do not retire each image once its instance is graded; "
                        "faster to re-grade, and about 1.7 GB per instance")
    p.add_argument("--force", action="store_true",
                   help="re-grade instances already in eval_results.jsonl")
    args = p.parse_args(argv)
    args.predictions = args.predictions.resolve()
    args.udocker_dir = args.udocker_dir.resolve()
    if not args.run_id:
        args.run_id = args.predictions.parent.name
    args.log_dir = args.predictions.parent / "eval_logs"
    args.results_log = args.predictions.parent / "eval_results.jsonl"
    args.workspaces = Path(tempfile.gettempdir()) / f"swebench-grade-{args.run_id}"
    args.retire_images = not args.keep_images
    args.disk = DiskGate(args.min_free_gb)
    args.runtime_impl = build_runtime(args.runtime, RuntimeConfig(
        udocker=args.udocker,
        udocker_dir=args.udocker_dir,
        network=args.network,
        pull_timeout=args.pull_timeout,
        copy_timeout=args.copy_timeout,
    ))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.predictions.exists():
        raise SystemExit(f"no predictions at {args.predictions}")
    if not harness_installed():
        raise SystemExit(
            "the swebench package is not importable. Install it with:\n"
            f"  pip install -e {BRIDGE_DIR.parent / 'SWE-bench'}")
    why = args.runtime_impl.preflight()
    if why:
        raise SystemExit(f"--runtime {args.runtime} is unusable: {why}")

    from datasets import load_dataset

    # Two different sets, and conflating them is how a re-grade of two flaky
    # instances silently rewrites a 30-instance report as a 2-instance one.
    # `predictions` is what the run submitted and is what the report describes;
    # `wanted` only narrows what gets a container this time.
    predictions = load_predictions(args.predictions)
    if not predictions:
        raise SystemExit("no predictions to grade")
    wanted = set(args.instance_ids) if args.instance_ids else set(predictions)
    unknown = wanted - set(predictions)
    if unknown:
        raise SystemExit(f"not in {args.predictions}: {', '.join(sorted(unknown))}")

    rows = {r["instance_id"]: dict(r)
            for r in load_dataset(args.dataset, split=args.split)}
    missing = set(predictions) - set(rows)
    if missing:
        raise SystemExit(f"not in {args.dataset}: {', '.join(sorted(missing))}")

    results: dict[str, dict] = load_done(args.results_log)
    results = {k: v for k, v in results.items() if k in predictions}
    if args.force:
        # Forget only what was asked for. A --force limited to two instances is
        # a re-grade of those two, not a discard of the other twenty-eight.
        results = {k: v for k, v in results.items() if k not in wanted}
    todo = [iid for iid in predictions if iid in wanted and iid not in results]
    if results:
        log(f"resuming: {len(results)} verdict(s) already in "
            f"{args.results_log.name}")
    if not todo:
        log("nothing left to grade")
    model_name = next(iter(predictions.values()))["model_name_or_path"]
    args.workspaces.mkdir(parents=True, exist_ok=True)

    began = time.time()
    if todo:
        log(f"grading {len(todo)} instance(s) on {args.runtime}, "
            f"{args.workers} worker(s), {free_gb():.1f}G free")
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(grade_one, args, args.runtime_impl,
                            rows[iid], predictions[iid]): iid
                for iid in todo
            }
            for fut in as_completed(futures):
                iid = futures[fut]
                try:
                    entry = fut.result()
                except Exception as e:  # noqa: BLE001 - one instance must not end it
                    entry = {iid: {"patch_is_None": False, "patch_exists": True,
                                   "patch_successfully_applied": False,
                                   "resolved": False,
                                   "error": f"{type(e).__name__}: {e}"}}
                results[iid] = entry
                record_verdict(args.results_log, iid, entry)
                verdict = "RESOLVED" if entry.get(iid, {}).get("resolved") else "no"
                log(f"[{iid}] {verdict}")

    report = build_report(predictions, results, model_name)
    out = report_path_for(args.predictions, model_name, args.run_id)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.rmtree(args.workspaces, ignore_errors=True)

    log(f"\n{report['resolved_instances']}/{report['submitted_instances']} resolved "
        f"in {round(time.time() - began)}s"
        f"\n  report : {out}"
        f"\n  logs   : {args.log_dir}"
        f"\n  free   : {free_gb():.1f}G")
    return 0


if __name__ == "__main__":
    sys.exit(main())
