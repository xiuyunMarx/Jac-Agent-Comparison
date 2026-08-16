#!/usr/bin/env python3
"""Generate SWE-bench predictions with one coding-agent implementation.

    python run_agent.py --framework openai --run-id smoke --limit 2

`--framework` picks the agent; see frameworks.py for the registry. Everything
else here is shared on purpose: every implementation gets the same workspace,
the same container, the same objective text, the same preparation step and the
same patch extraction, so a difference in the score is a difference between the
agents rather than between drivers that drifted apart.

Inference only. This produces `predictions.jsonl`; grading is grade.py's job,
which re-runs everything in a clean container from the instance image, so
nothing done here can flatter the score.

One instance is one workspace and one container:

  1. the instance image's /testbed is copied onto the host, giving a tree at
     base_commit that already carries the compiled artifacts and the .egg-info
     the image's conda env is installed against;
  2. that tree is bind-mounted back into a running container at /testbed;
  3. the agent reads and writes the host copy with its ordinary file tools,
     while its run_command goes through `docker exec` into the container
     (CODEAGENT_EXEC=docker), so `pytest` is the instance's pytest;
  4. the patch is whatever the workspace gained relative to the tree we handed
     it, which is not the same thing as `git diff` -- see workspace.py.

No agent is imported into this process. The byLLM one cannot be -- the `jac` CLI
bundles its own interpreter -- and the Python ones are held to the same shape so
all three are isolated identically: one instance is one process, and a crash
costs that instance rather than the run. stdout is not a channel either; byLLM,
litellm, LangChain and httpx all log to it. swe_entry.jac and swe_entry.py are
the shims: job file in, result file out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import frameworks
import workspace as ws_mod
from runtime import (
    RUNTIMES,
    DiskGate,
    RuntimeConfig,
    StepError,
    build_runtime,
    git,
    log,
    run,
)

BRIDGE_DIR = Path(__file__).resolve().parent

# Any one of these is enough; the agents read whichever their provider needs.
PROVIDER_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_API_KEY")

_write_lock = threading.Lock()


# --------------------------------------------------------------------------
# The agent process
# --------------------------------------------------------------------------


def agent_env(args: argparse.Namespace, container: str) -> dict[str, str]:
    env = dict(os.environ)
    env["CODEAGENT_HOME"] = str(args.agent_home)
    env["CODEAGENT_MODEL"] = args.model
    env["CODEAGENT_EXEC"] = args.exec_backend
    env["CODEAGENT_EXEC_CONTAINER"] = container
    if args.exec_backend == "udocker":
        env["CODEAGENT_UDOCKER"] = args.udocker
        env["UDOCKER_DIR"] = str(args.udocker_dir)
    env["CODEAGENT_EXEC_WORKDIR"] = "/testbed"
    if args.exec_user:
        env["CODEAGENT_EXEC_USER"] = args.exec_user
    env["PYTHONUNBUFFERED"] = "1"
    return env


def runner_bin(args: argparse.Namespace) -> str:
    return args.jac if args.fw.runner == "jac" else args.python


def spawn_agent(args: argparse.Namespace, inst: dict, ws: Path,
                container: str, work: Path) -> dict:
    job = work / "job.json"
    result = work / "result.json"
    # A --force re-run reuses this directory, and a stale result file would be
    # read back as if this attempt had produced it.
    result.unlink(missing_ok=True)
    job.write_text(json.dumps({
        "instance_id": inst["instance_id"],
        "objective": ws_mod.build_objective(inst),
        "repo_root": str(ws),
        "max_steps": args.max_steps,
    }, indent=2), encoding="utf-8")

    argv = args.fw.argv(runner_bin(args), str(job), str(result))
    try:
        done = run(argv, cwd=args.agent_home, env=agent_env(args, container),
                   timeout=args.instance_timeout, check=False)
        transcript = (done.stdout or "") + (done.stderr or "")
        exit_note = ("" if done.returncode == 0
                     else f"{args.fw.runner} exited {done.returncode}")
    except StepError as e:
        transcript = ""
        exit_note = str(e)
    (work / "agent.log").write_text(transcript, encoding="utf-8")

    if result.exists():
        record = json.loads(result.read_text(encoding="utf-8"))
        if exit_note and not record.get("error"):
            record["error"] = exit_note
        return record
    # No result file: the agent died before it could write one. Its edits are
    # still on disk, so the caller goes on to collect whatever patch exists.
    return {
        "instance_id": inst["instance_id"],
        "error": exit_note or "the agent produced no result file",
        "steps": 0, "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "cached_tokens": 0, "tool_calls": [], "tool_call_count": 0,
        "wall_clock_sec": 0.0,
    }


def warm_agent(args: argparse.Namespace) -> None:
    """Load the agent once, before the workers can do it simultaneously.

    For byLLM this is a real race: every worker runs `jac run` from the same
    project and the compiled-module cache under .jac/ is shared, so letting N of
    them compile the same modules at once is a fight over the same cache files.
    One warm run makes the rest hits.

    For the Python sides it is a fail-fast check instead -- an unimportable agent
    or a missing dependency surfaces here rather than 300 containers later.

    Either way: invoked with no arguments a shim prints its usage and exits
    non-zero, after importing the whole agent, which is the part being warmed.
    """
    done = run(args.fw.argv(runner_bin(args)), cwd=args.agent_home,
               env=agent_env(args, ""), check=False, timeout=600)
    if "usage:" not in (done.stdout or "") + (done.stderr or ""):
        log("warning: the agent did not load cleanly during warm-up:\n"
            + ((done.stderr or done.stdout or "").strip()[:800] or "(no output)"))


# --------------------------------------------------------------------------
# One instance, end to end
# --------------------------------------------------------------------------


def solve_instance(args: argparse.Namespace, inst: dict) -> dict:
    iid = inst["instance_id"]
    began = time.time()
    ws = args.workspaces / iid
    work = args.logs / iid
    work.mkdir(parents=True, exist_ok=True)
    runtime = args.runtime_impl
    # Framework-prefixed: a comparison run has several agents in flight over the
    # same instance ids, and a shared container name would have them fight.
    container = f"{args.framework}-swe-{iid}-{args.run_id}"[:120]
    record: dict = {
        "instance_id": iid,
        "repo": inst["repo"],
        "framework": args.framework,
        "model": args.model,
        "patch": "",
        "error": "",
    }
    # Accumulated rather than assigned: preparation can go wrong before the
    # agent runs, and the agent's own error must not overwrite that.
    problems: list[str] = []

    def note(problem: str) -> None:
        problems.append(problem)
        log(f"[{iid}] {problem}")

    started_container = False
    baseline: set[tuple[str, str]] = set()
    prepared = False
    try:
        args.disk.wait(iid)
        runtime.ensure_image(inst["image"])
        log(f"[{iid}] preparing workspace")
        # Under udocker the container is the workspace, so this returns the path
        # inside it rather than the one the docker path would have filled.
        ws = runtime.create(container, inst["image"], ws)
        started_container = True
        head = git(ws, "rev-parse", "HEAD").strip()
        if head != inst["base_commit"]:
            # Trust the dataset over the image, and say so: a silent reset here
            # would hide an image/dataset mismatch that invalidates the run.
            log(f"[{iid}] image HEAD {head[:12]} != base "
                f"{inst['base_commit'][:12]}; resetting")
            git(ws, "checkout", "--force", inst["base_commit"])
        baseline = ws_mod.porcelain(ws)
        prepared = True
        if not runtime.alive(container):
            raise StepError(f"the workspace container {container} did not stay up")
        if ws_mod.preparation_script(inst, args.prepare):
            log(f"[{iid}] wiring the environment to the workspace")
            trouble = ws_mod.prepare_workspace(
                runtime, container, inst, args.prepare, args.install_timeout)
            if trouble:
                # Not fatal, but recorded: the failure mode it leaves behind is
                # the agent testing against code it did not write, which looks
                # like a confused agent rather than a broken workspace.
                note(f"workspace preparation: {trouble}")

        log(f"[{iid}] running agent")
        outcome = spawn_agent(args, inst, ws, container, work)
        if outcome.get("error"):
            note(str(outcome["error"]))
        record.update({k: v for k, v in outcome.items()
                       if k not in ("instance_id", "error")})
    except StepError as e:
        note(str(e))
    except Exception as e:  # noqa: BLE001 - one instance must not end the run
        note(f"{type(e).__name__}: {e}")
    finally:
        if started_container:
            # Before anything on the host touches the workspace: under docker
            # the commands ran as real root, so their leavings are root-owned
            # inside the bind mount. Under udocker this is a no-op.
            runtime.reclaim(container, ws)

    # Separate from the block above on purpose. An agent that crashed partway
    # still left edits on disk, and those are worth grading.
    if prepared:
        try:
            record["patch"] = ws_mod.extract_patch(ws, baseline)
        except StepError as e:
            note(f"patch extraction failed: {e}")
        except Exception as e:  # noqa: BLE001
            note(f"patch extraction failed: {type(e).__name__}: {e}")

    # After the patch is taken, never before: under udocker this removes the
    # container, and the workspace lives inside it.
    if started_container:
        runtime.destroy(container, ws, args.keep_workspaces)
    if args.cleanup_images:
        runtime.remove_image(inst["image"])

    record["error"] = "; ".join(problems)
    record["total_sec"] = round(time.time() - began, 2)
    record["patch_bytes"] = len(record["patch"].encode("utf-8"))
    log(f"[{iid}] done in {record['total_sec']}s, "
        f"patch {record['patch_bytes']}B"
        + (f", error: {record['error'][:120]}" if record["error"] else ""))
    return record


def append(path: Path, row: dict) -> None:
    with _write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def load_instances(args: argparse.Namespace) -> list[dict]:
    from datasets import load_dataset

    rows = [dict(r) for r in load_dataset(args.dataset, split=args.split)]
    if args.instance_ids:
        wanted = set(args.instance_ids)
        missing = wanted - {r["instance_id"] for r in rows}
        if missing:
            raise SystemExit(f"not in {args.dataset}: {', '.join(sorted(missing))}")
        rows = [r for r in rows if r["instance_id"] in wanted]
    if args.repo:
        rows = [r for r in rows if r["repo"] in set(args.repo)]
    rows.sort(key=lambda r: r["instance_id"])
    if args.limit:
        rows = rows[:args.limit]
    return rows


def add_selection_args(p: argparse.ArgumentParser) -> None:
    """The flags that choose instances. Shared with compare.py, which owns them
    for every side at once so the sides cannot end up on different sets."""
    p.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
    p.add_argument("--split", default="test")
    p.add_argument("--instance-ids", nargs="*", default=[],
                   help="run only these instances")
    p.add_argument("--instances-file", type=Path, default=None,
                   help="a file of instance ids, one per line (# comments ok); "
                        "merged with --instance-ids")
    p.add_argument("--repo", nargs="*", default=[],
                   help="run only instances from these repos, e.g. psf/requests")
    p.add_argument("--limit", type=int, default=0,
                   help="run only the first N instances after filtering")


def resolve_instance_ids(args: argparse.Namespace) -> list[str]:
    ids = list(args.instance_ids)
    if args.instances_file:
        if not args.instances_file.exists():
            raise SystemExit(f"no instance list at {args.instances_file}")
        for line in args.instances_file.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                ids.append(line)
    # Deduplicated but order-independent: load_instances sorts anyway, and a
    # duplicate id would otherwise make --limit mean fewer instances than it says.
    return sorted(set(ids))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one coding-agent implementation over SWE-bench instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--framework", default="byllm", choices=frameworks.NAMES,
                   help="; ".join(f"{n}: {frameworks.FRAMEWORKS[n].blurb}"
                                  for n in frameworks.ORDER))
    add_selection_args(p)
    p.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"),
                   help="names the output directory and the containers")
    p.add_argument("--output-dir", type=Path, default=BRIDGE_DIR / "results")
    p.add_argument("--model", default=os.environ.get("CODEAGENT_MODEL", "gpt-5"),
                   help="pin it identically across the frameworks, "
                        "or you are measuring the model")
    p.add_argument("--model-name", default="",
                   help="model_name_or_path in predictions.jsonl "
                        "(default: <framework>-codeagent/<model>)")
    p.add_argument("--max-steps", type=int, default=10,
                   help="phase budget for one agent run")
    p.add_argument("--workers", type=int, default=4,
                   help="instances in flight at once; each holds a container")
    p.add_argument("--agent-home", type=Path, default=None,
                   help="default: the --framework's own directory")
    p.add_argument("--jac", default="jac",
                   help="the jac CLI to run the byllm agent with")
    p.add_argument("--python", default=sys.executable,
                   help="the interpreter to run a Python agent with; it needs "
                        "that project's dependencies importable")
    p.add_argument("--runtime", default="udocker", choices=sorted(RUNTIMES),
                   help="how to get the instance image and run commands in it. "
                        "'udocker' needs no daemon, no root and no docker group")
    p.add_argument("--udocker", default=os.environ.get("CODEAGENT_UDOCKER", "udocker"))
    p.add_argument("--udocker-dir", type=Path,
                   default=Path(os.environ.get("UDOCKER_DIR",
                                               Path.home() / ".udocker")),
                   help="udocker's image and container repository; "
                        "images are 1-2 GB each")
    p.add_argument("--exec-backend", default="",
                   help="where the agent's own commands run: defaults to the "
                        "--runtime, or 'local' to run them on this machine, "
                        "which for these repos means unusable test feedback")
    p.add_argument("--exec-user", default="",
                   help="user for commands in the container; default is root, "
                        "as in the official harness")
    p.add_argument("--network", default="",
                   help="docker network for the workspace container, "
                        "e.g. 'none' to cut it off")
    p.add_argument("--prepare", default="auto", choices=["auto", "always", "never"],
                   help="run the instance's own install step in the container. "
                        "'auto' runs only what the bind mount actually breaks -- "
                        "a non-editable install, plus cheap setup like locale-gen")
    p.add_argument("--install-timeout", type=float, default=900)
    p.add_argument("--instance-timeout", type=float, default=1800,
                   help="wall-clock ceiling for one agent run")
    p.add_argument("--pull-timeout", type=float, default=3600)
    p.add_argument("--copy-timeout", type=float, default=1800)
    p.add_argument("--min-free-gb", type=float, default=6.0,
                   help="a worker waits until the disk has this much room "
                        "before unpacking another container")
    p.add_argument("--keep-workspaces", action="store_true",
                   help="keep each workspace after extracting its patch "
                        "(a full repo copy per instance)")
    p.add_argument("--cleanup-images", action="store_true",
                   help="delete each instance image once done; trades bandwidth "
                        "for disk on long runs")
    p.add_argument("--allow-no-key", action="store_true",
                   help="start even with no provider API key set; only "
                        "useful for exercising the harness itself")
    p.add_argument("--force", action="store_true",
                   help="re-run instances already present in predictions.jsonl")
    args = p.parse_args(argv)
    return finalize(args)


def finalize(args: argparse.Namespace) -> argparse.Namespace:
    """Derive everything the driver reads. Split out so compare.py can build an
    equivalent namespace without going through this parser twice."""
    args.fw = frameworks.get(args.framework)
    args.instance_ids = resolve_instance_ids(args)
    args.udocker_dir = args.udocker_dir.resolve()
    # The agent executes commands through whichever runtime holds the container,
    # unless explicitly overridden. Keeping these in step matters: a `docker
    # exec` into a container that only udocker knows about finds nothing.
    if not args.exec_backend:
        args.exec_backend = args.runtime
    if args.exec_backend not in ("docker", "udocker", "local"):
        raise SystemExit(f"unknown --exec-backend: {args.exec_backend}")
    args.runtime_impl = build_runtime(args.runtime, RuntimeConfig(
        udocker=args.udocker,
        udocker_dir=args.udocker_dir,
        network=args.network,
        pull_timeout=args.pull_timeout,
        copy_timeout=args.copy_timeout,
    ))
    args.disk = DiskGate(args.min_free_gb)
    args.agent_home = (args.agent_home or args.fw.home).resolve()
    args.out = (args.output_dir / args.run_id).resolve()
    args.workspaces = args.out / "workspaces"
    args.logs = args.out / "logs"
    args.predictions = args.out / "predictions.jsonl"
    args.runs = args.out / "runs.jsonl"
    if not args.model_name:
        args.model_name = f"{args.framework}-codeagent/{args.model}"
    return args


def preflight(args: argparse.Namespace) -> None:
    why = args.fw.check(args.agent_home, runner_bin(args))
    if why:
        raise SystemExit(why)
    # The runtime is needed whatever --exec-backend says: the workspace itself
    # comes out of the instance image.
    why = args.runtime_impl.preflight()
    if why:
        raise SystemExit(
            f"--runtime {args.runtime} is unusable: {why}"
            + ("\n\nNo docker group on this machine? Use --runtime udocker, "
               "which needs neither a daemon nor root."
               if args.runtime == "docker" else ""))
    if not any(os.environ.get(k) for k in PROVIDER_KEYS):
        # Fatal, not a warning. Without a key every instance still pulls its
        # image, unpacks a container and runs the preparation step before dying
        # at the first LLM call -- so the run looks busy for an hour, writes a
        # full set of empty-patch predictions, and those then satisfy the resume
        # check on the next attempt. Stopping here costs a second; not stopping
        # costs the run twice.
        if not args.allow_no_key:
            raise SystemExit(
                "no provider API key in the environment. Set one of: "
                + ", ".join(PROVIDER_KEYS) + "\n"
                "  export OPENAI_API_KEY=sk-...\n\n"
                "Every instance would otherwise fail at its first LLM call and "
                "record an empty patch. Pass --allow-no-key to run anyway, for "
                "exercising the harness itself.")
        log("warning: --allow-no-key and no key is set; "
            "every instance will fail at the first LLM call")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    preflight(args)

    instances = load_instances(args)
    for d in (args.out, args.workspaces, args.logs):
        d.mkdir(parents=True, exist_ok=True)

    if args.predictions.exists() and not args.force:
        done = {json.loads(line)["instance_id"]
                for line in args.predictions.read_text(encoding="utf-8").splitlines()
                if line.strip()}
        instances = [i for i in instances if i["instance_id"] not in done]
        if done:
            log(f"resuming: {len(done)} instance(s) already predicted")

    if not instances:
        log("nothing to do")
        return 0

    warm_agent(args)

    log(f"run {args.run_id}: {args.framework} on {args.runtime}, "
        f"{len(instances)} instance(s), model {args.model}, "
        f"{args.workers} worker(s) -> {args.out}")
    began = time.time()
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(solve_instance, args, i): i for i in instances}
        for fut in as_completed(futures):
            record = fut.result()
            records.append(record)
            append(args.predictions, {
                "instance_id": record["instance_id"],
                "model_name_or_path": args.model_name,
                "model_patch": record["patch"],
            })
            append(args.runs, {k: v for k, v in record.items() if k != "patch"})

    empty = sum(1 for r in records if not r["patch"])
    failed = sum(1 for r in records if r["error"])
    tokens = sum(int(r.get("prompt_tokens", 0) or 0)
                 + int(r.get("completion_tokens", 0) or 0) for r in records)
    log(
        f"\n{len(records)} instance(s) in {round(time.time() - began)}s"
        f"\n  patches produced : {len(records) - empty}"
        f"\n  empty patches    : {empty}"
        f"\n  runs with errors : {failed}"
        f"\n  tokens           : {tokens:,}"
        f"\n  predictions      : {args.predictions}"
        f"\n\nGrade them with:"
        f"\n  python {BRIDGE_DIR / 'grade.py'} --predictions {args.predictions}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
