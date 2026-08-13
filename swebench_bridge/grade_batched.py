#!/usr/bin/env python3
"""Grade predictions on a filesystem that has no room for the image cache.

grade_local.py assumes it can hold every instance image at once -- ~96G for
SWE-bench Lite -- and keeps its verdicts in memory until the last one lands.
On a full disk both assumptions fail together: containers unpack incompletely,
every verdict turns to garbage, and a crash at instance 143 loses all 143.

This driver keeps grade_local's logic and changes only the bookkeeping:

  * one image at a time is retired. `udocker rmi` after an instance is graded
    frees that instance's exclusive layers (~340MB on Lite), so free space
    climbs as the run proceeds rather than falling. Nothing is re-downloaded
    that was not deliberately evicted first.
  * a worker waits for MIN_FREE_GB before it unpacks anything, so a container
    is never written into a filesystem that cannot hold it. That is the exact
    failure that voided the previous two passes.
  * every verdict is appended to eval_results.jsonl the moment it is known.
    Re-running skips what is already there, so a crash costs one instance.

The verdicts themselves still come from grade_local.grade_one, which still
gets them from swebench's own make_test_spec/get_eval_report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BRIDGE = Path(__file__).resolve().parent
sys.path.insert(0, str(BRIDGE))

import grade_local  # noqa: E402
from run_agent import log  # noqa: E402

MIN_FREE_GB = 6.0      # a container must fit before it is unpacked
EVICT_TARGET_GB = 14.0  # headroom to clear before the first instance starts
_disk_lock = threading.Lock()


def free_gb(path: Path = Path.home()) -> float:
    st = shutil.disk_usage(path)
    return st.free / 2**30


def image_of(instance: dict) -> str:
    return grade_local.swebench_bits()[0](instance).image


def rmi(runtime, image: str) -> None:
    """Retire one image. Shared base layers survive until their last user goes."""
    try:
        runtime._udocker("rmi", image, check=False, timeout=300)
    except Exception as e:  # noqa: BLE001 - reclaiming space must not end the run
        log(f"  (rmi {image} failed: {type(e).__name__}: {e})")


def wait_for_room(iid: str) -> None:
    """Block until the filesystem can hold a container, reporting once."""
    warned = False
    while free_gb() < MIN_FREE_GB:
        if not warned:
            log(f"[{iid}] waiting for room ({free_gb():.1f}G free, need {MIN_FREE_GB}G)")
            warned = True
        time.sleep(20)


def evict_for_headroom(runtime, order: list[str], rows: dict) -> int:
    """Drop the images of the instances scheduled last, to make a start possible.

    They are re-pulled when their turn comes, by which point the instances
    graded before them will have freed far more than they take.
    """
    dropped = 0
    for iid in reversed(order):
        if free_gb() >= EVICT_TARGET_GB:
            break
        rmi(runtime, image_of(rows[iid]))
        dropped += 1
    if dropped:
        log(f"evicted {dropped} image(s) scheduled last; {free_gb():.1f}G free")
    return dropped


def grade_and_retire(args, runtime, instance: dict, pred: dict) -> dict:
    iid = instance["instance_id"]
    wait_for_room(iid)
    try:
        return grade_local.grade_one(args, runtime, instance, pred)
    finally:
        with _disk_lock:
            rmi(runtime, image_of(instance))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
    p.add_argument("--split", default="test")
    p.add_argument("--run-id", default="")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--instance-ids", nargs="*", default=[])
    p.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    cli = p.parse_args(argv)
    globals()["MIN_FREE_GB"] = cli.min_free_gb

    # grade_local owns the argument surface its own code reads; borrow it whole
    # so this driver cannot drift from the grader it delegates to.
    args = grade_local.parse_args([
        "--predictions", str(cli.predictions),
        "--dataset", cli.dataset, "--split", cli.split,
        "--runtime", "udocker", "--workers", str(cli.workers),
        "--timeout", str(cli.timeout),
        *(["--run-id", cli.run_id] if cli.run_id else []),
    ])
    runtime = args.runtime_impl
    why = runtime.preflight()
    if why:
        raise SystemExit(f"udocker is unusable: {why}")

    from datasets import load_dataset

    preds = grade_local.load_predictions(args.predictions)
    if cli.instance_ids:
        preds = {k: v for k, v in preds.items() if k in set(cli.instance_ids)}
    rows = {r["instance_id"]: dict(r)
            for r in load_dataset(args.dataset, split=args.split)}

    ledger = args.predictions.parent / "eval_results.jsonl"
    done: dict[str, dict] = {}
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["instance_id"]] = rec["entry"]
        log(f"resuming: {len(done)} instance(s) already graded")

    todo = [i for i in preds if i not in done]
    if not todo:
        log("nothing left to grade")
    args.workspaces.mkdir(parents=True, exist_ok=True)
    log(f"grading {len(todo)} instance(s), {cli.workers} worker(s), "
        f"{free_gb():.1f}G free")
    evict_for_headroom(runtime, todo, rows)

    began = time.time()
    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, cli.workers)) as pool:
        futs = {pool.submit(grade_and_retire, args, runtime, rows[i], preds[i]): i
                for i in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            iid = futs[fut]
            try:
                entry = fut.result()
            except Exception as e:  # noqa: BLE001 - one instance must not end the run
                entry = {iid: {"patch_is_None": False, "patch_exists": True,
                               "patch_successfully_applied": False,
                               "resolved": False, "error": f"{type(e).__name__}: {e}"}}
            done[iid] = entry
            with write_lock:
                with ledger.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"instance_id": iid, "entry": entry}) + "\n")
            ok = entry.get(iid, {}).get("resolved")
            log(f"[{n}/{len(todo)}] {iid} {'RESOLVED' if ok else 'no'} "
                f"({free_gb():.1f}G free)")

    model = next(iter(preds.values()))["model_name_or_path"]
    report = grade_local.build_report(preds, {i: done[i] for i in preds if i in done},
                                      model)
    out = args.predictions.parent / f"{model.replace('/', '__')}.{args.run_id}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.rmtree(args.workspaces, ignore_errors=True)
    log(f"\n{report['resolved_instances']}/{report['submitted_instances']} resolved "
        f"in {round(time.time() - began)}s\n  report : {out}\n  ledger : {ledger}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
