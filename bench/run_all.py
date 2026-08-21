"""The sweep: five benchmarks, three arms each, one model, one command.

Every project here already has a runner that knows how to drive its own arms --
`compare.py` pins an instance set across three frameworks, `e2e.py` brings up a
database before it runs anything, `run_eval.py` resumes a half-finished sweep.
This module does not reimplement any of that. It decides what runs in what
order, hands each runner the environment that points it at the shared local
model, and records what happened.

Stages run cheapest-first, so a broken model seam shows up in the two-minute
benchmark rather than four hours into SWE-bench. Within a stage the arms run
one at a time, because they are being compared on tokens and wall-clock.

    python -m bench.run_all                      # everything
    python -m bench.run_all --smoke              # a few cases per benchmark
    python -m bench.run_all --only meeting,email
    python -m bench.run_all --skip codeagent
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import config

R = config.REPO_ROOT
PY = sys.executable


@dataclass
class Step:
    """One subprocess, with the environment that points it at the local model."""
    name: str
    argv: list[str]
    project: str
    cwd: Path | None = None
    extra_env: dict[str, str] = field(default_factory=dict)
    #: A stage's scoring step should still run when its agent step reported a
    #: non-zero exit -- a partial run is worth scoring, and the exit code is in
    #: the manifest either way.
    required: bool = True


@dataclass
class Stage:
    key: str
    title: str
    steps: list[Step]


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

def stage_meeting(args) -> Stage:
    run = [PY, str(R / "meeting-assistant" / "eval" / "run.py"),
           "--impl", "byLLM", "--impl", "CrewAI", "--impl", "openai_sdk"]
    if args.smoke:
        run += ["--cases", "meeting_001", "--repeat", "1"]
    else:
        run += ["--repeat", str(args.repeats)]
    score = [PY, str(R / "meeting-assistant" / "eval" / "score.py"),
             str(R / "meeting-assistant" / "eval" / "runs")]
    if args.judge:
        score += ["--judge", "--judge-model", config.judge_model()]
    return Stage("meeting", "meeting-assistant", [
        Step("run", run, "meeting"),
        # One invocation over every arm: score.py rebuilds summary.json from
        # only the runs it is handed.
        Step("score", score, "meeting", required=False),
    ])


def stage_email(args) -> Stage:
    run = [PY, str(R / "Email-Auto-response" / "eval" / "run.py")]
    if args.smoke:
        run += ["--batches", "batch_001"]
    if args.judge:
        run += ["--judge"]
    # run.py scores every arm itself, in one invocation, for the same reason.
    return Stage("email", "Email-Auto-response", [Step("run+score", run, "email")])


def stage_ytnavigator(args) -> Stage:
    cmd = [PY, str(R / "YTNavigator" / "eval" / "e2e.py"), "--impl", "all"]
    if args.smoke:
        # Stops after the retrieval sanity check: brings up Postgres, builds the
        # dataset, proves retrieval works, and makes no LLM call. It verifies the
        # plumbing rather than the arms, which is what a smoke run is for here --
        # this benchmark has no per-question limit to cut a real run down with.
        cmd += ["--smoke", "--fake-embeddings"]
    elif args.judge:
        cmd += ["--judge", "--judge-model", config.litellm_id(config.judge_model())]
    return Stage("ytnavigator", "YTNavigator", [Step("e2e", cmd, "ytnavigator")])


def stage_raggpt(args) -> Stage:
    harness = R / "RagGPT" / "eval" / "harness"
    run = [PY, str(harness / "run_eval.py"),
           "--systems", "langgraph,jac,jac-byllm-router,openai-sdk"]
    run += ["--repeats", "1" if args.smoke else str(args.repeats)]
    if args.smoke:
        run += ["--limit", "5"]
    return Stage("raggpt", "RagGPT", [
        Step("run", run, "raggpt", cwd=harness),
        Step("score", [PY, str(harness / "score.py")], "raggpt", cwd=harness, required=False),
        Step("report", [PY, str(harness / "report.py")], "raggpt", cwd=harness, required=False),
    ])


def stage_codeagent(args) -> Stage:
    bridge = R / "CodeAgent" / "swebench_bridge"
    study = R / "CodeAgent" / "case_study"
    run_id = args.run_id
    compare = [PY, str(bridge / "compare.py"),
               "--run-id", run_id,
               "--frameworks", "byllm", "langgraph", "openai",
               "--workers", str(args.workers),
               "--eval-workers", str(args.eval_workers)]
    if args.smoke:
        compare += ["--limit", "2"]
    else:
        # The pinned set, so this run is comparable with the last one. Passing
        # it to compare.py (not through to run_agent) is what guarantees every
        # framework gets the identical instances.
        compare += ["--instances-file", str(study / "instances.txt")]
    build = [PY, str(study / "build_study.py"), "--out", run_id] + [
        str(bridge / "results" / f"{run_id}-{fw}") for fw in ("byllm", "langgraph", "openai")]
    return Stage("codeagent", "CodeAgent (SWE-bench Lite)", [
        Step("compare", compare, "codeagent", cwd=bridge),
        Step("study", build, "codeagent", cwd=study, required=False),
    ])


STAGE_BUILDERS = {
    "meeting": stage_meeting,
    "email": stage_email,
    "ytnavigator": stage_ytnavigator,
    "raggpt": stage_raggpt,
    "codeagent": stage_codeagent,
}
#: Cheapest first. CodeAgent is hours and millions of tokens; it goes last so a
#: broken seam is found in minutes.
STAGE_ORDER = ["meeting", "email", "ytnavigator", "raggpt", "codeagent"]


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def mark(project: str, step_name: str) -> None:
    """Attribute the next calls to this step in the shared token ledger.

    proxy.py tags every logged call with whatever marker was last posted. RagGPT's
    runner re-marks per turn (it needs per-item attribution); the other four
    projects never mark at all, so without this their calls would all land under
    a null system and the ledger could say nothing about who spent what.
    """
    if not config.use_proxy():
        return
    url = f"http://127.0.0.1:{config.proxy_port()}/__mark"
    payload = json.dumps({"system": project, "item_id": step_name,
                          "repeat": 0, "turn": 0, "attempt": f"{project}:{step_name}"})
    req = urllib.request.Request(url, data=payload.encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:
        # No proxy is a supported configuration (BENCH_NO_PROXY=1), and a
        # missing ledger must never stop a benchmark that is otherwise fine.
        pass


def run_step(step: Step, log_dir: Path, dry_run: bool) -> dict:
    env = config.env_for(step.project)
    env.update(step.extra_env)
    printable = " ".join(step.argv)
    if dry_run:
        print(f"    would run: {printable}")
        return {"step": step.name, "argv": step.argv, "skipped": "dry-run"}

    mark(step.project, step.name)
    log_path = log_dir / f"{step.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"    $ {printable}")
    print(f"      log: {log_path}")
    t0 = time.perf_counter()
    with log_path.open("w") as log:
        log.write(f"$ {printable}\n\n")
        log.flush()
        proc = subprocess.Popen(step.argv, cwd=str(step.cwd) if step.cwd else None,
                                env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            # The console gets the tail; the log gets everything.
            sys.stdout.write(f"      | {line}" if len(line) < 400 else f"      | {line[:400]}...\n")
        code = proc.wait()
    elapsed = time.perf_counter() - t0
    print(f"      -> exit {code} in {elapsed:.1f}s")
    return {"step": step.name, "argv": step.argv, "exit_code": code,
            "seconds": round(elapsed, 1), "log": str(log_path)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"),
                    help="names this run's log directory and CodeAgent's result dirs")
    ap.add_argument("--only", default="", help="comma-separated stages to run")
    ap.add_argument("--skip", default="", help="comma-separated stages to skip")
    ap.add_argument("--smoke", action="store_true",
                    help="a few cases per benchmark: proves the wiring, not the model")
    ap.add_argument("--repeats", type=int, default=3,
                    help="repetitions where the runner supports them (default: 3)")
    ap.add_argument("--judge", dest="judge", action="store_true", default=True,
                    help="LLM-judged quality, on the same local model (default: on)")
    ap.add_argument("--no-judge", dest="judge", action="store_false",
                    help="deterministic metrics only")
    ap.add_argument("--workers", type=int, default=4, help="CodeAgent instances in flight")
    ap.add_argument("--eval-workers", type=int, default=2, help="CodeAgent grading workers")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--skip-jac-check", action="store_true",
                    help="run even though the Jac runtime is unusable (byLLM arms will fail)")
    args = ap.parse_args(argv)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    for name in only | skip:
        if name not in STAGE_BUILDERS:
            raise SystemExit(f"unknown stage {name!r}; known: {', '.join(STAGE_ORDER)}")
    selected = [k for k in STAGE_ORDER if (not only or k in only) and k not in skip]
    if not selected:
        raise SystemExit("no stages selected")

    # Every one of the five benchmarks has a byLLM arm, so a jac that is missing
    # or incompatible does not degrade the sweep -- it removes one of the three
    # things being compared, and does it as an identical unhelpful line once per
    # run. Check once, up front, and say what is wrong.
    if not args.dry_run and not args.skip_jac_check:
        from . import verify
        jac_rows = verify.check_jac_runtime()
        if any(row[2] == verify.FAIL for row in jac_rows):
            print("\nThe Jac runtime is not usable, so every byLLM arm would fail:\n")
            for bench, item, status, kind, detail in jac_rows:
                print(f"  {status:4}  {item:18s} {detail}")
            print("\nA three-way comparison missing one arm is not a comparison. Fix this\n"
                  "first (see BENCHMARK.md step 3), or pass --skip-jac-check to run the\n"
                  "other two arms anyway and accept the gap.")
            return 1

    run_dir = config.RUNS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    desc = config.describe()
    print("=" * 72)
    print(f"Benchmark sweep {args.run_id}")
    for key, value in desc.items():
        print(f"  {key:20s} {value}")
    print(f"  {'stages':20s} {', '.join(selected)}")
    print(f"  {'judge':20s} {'on (same local model)' if args.judge else 'off'}")
    print(f"  {'mode':20s} {'smoke' if args.smoke else 'full'}")
    print("=" * 72)

    manifest = {"run_id": args.run_id, "config": desc, "smoke": args.smoke,
                "judge": args.judge, "repeats": args.repeats, "stages": []}

    failures = 0
    for key in selected:
        stage = STAGE_BUILDERS[key](args)
        print(f"\n## {stage.title}")
        record = {"stage": key, "title": stage.title, "steps": []}
        stage_failed = False
        for step in stage.steps:
            if stage_failed and step.required:
                print(f"    skipping {step.name}: an earlier required step failed")
                record["steps"].append({"step": step.name, "skipped": "earlier failure"})
                continue
            result = run_step(step, run_dir / key, args.dry_run)
            record["steps"].append(result)
            if result.get("exit_code", 0) != 0 and step.required:
                stage_failed = True
        record["ok"] = not stage_failed
        if stage_failed:
            failures += 1
        manifest["stages"].append(record)
        if not args.dry_run:
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.dry_run:
        print("\n(dry run -- nothing was executed)")
        return 0

    print(f"\n{len(selected) - failures}/{len(selected)} stage(s) completed.")
    print(f"Logs and manifest: {run_dir}")
    print(f"Summarize with:  python -m bench.report --run-id {args.run_id}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
