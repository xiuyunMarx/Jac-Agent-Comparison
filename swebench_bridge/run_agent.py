#!/usr/bin/env python3
"""Generate SWE-bench predictions with either coding agent.

`--framework byllm` runs the Jac/byLLM agent in ../byLLM; `--framework
langgraph` runs the LangGraph agent in ../langgraph. Everything else in this
file is shared on purpose: the two frameworks get the same workspace, the same
container, the same objective text, the same preparation step and the same
patch extraction, so a difference in the score is a difference between the
agents rather than between two drivers that drifted apart. The only
framework-specific thing here is which shim gets spawned (see FRAMEWORKS).

Inference only. This produces `predictions.jsonl`; grading is the official
harness's job (see evaluate.py), which re-runs everything in a clean container
from the instance image, so nothing done here can flatter the score.

One instance is one workspace and one container:

  1. the instance image's /testbed is copied onto the host, giving a tree at
     base_commit that already carries the compiled artifacts and the .egg-info
     the image's conda env is installed against;
  2. that tree is bind-mounted back into a running container at /testbed;
  3. the agent reads and writes the host copy with its ordinary file tools,
     while its run_command goes through `docker exec` into the container
     (CODEAGENT_EXEC=docker), so `pytest` is the instance's pytest;
  4. the patch is whatever the workspace gained relative to the tree we handed
     it, which is not the same thing as `git diff` -- see extract_patch.

Neither agent is imported into this process. The byLLM one cannot be -- the
`jac` CLI bundles its own interpreter -- and the LangGraph one is held to the
same shape so both are isolated identically: one instance is one process, and a
crash costs that instance rather than the run. swe_entry.jac and swe_entry.py
are the two shims: job file in, result file out.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Framework:
    """Everything that differs between the two agents, which is very little.

    `marker` is the file that must exist under the agent home for it to be that
    agent at all, and `runner` names which interpreter option launches `entry`.
    """

    name: str
    home: Path
    entry: Path
    marker: str
    runner: str  # "jac" | "python"


FRAMEWORKS: dict[str, Framework] = {
    "byllm": Framework(
        name="byllm",
        home=BRIDGE_DIR.parent / "byLLM",
        entry=BRIDGE_DIR / "swe_entry.jac",
        marker="orchestrator.jac",
        runner="jac",
    ),
    "langgraph": Framework(
        name="langgraph",
        home=BRIDGE_DIR.parent / "langgraph",
        entry=BRIDGE_DIR / "swe_entry.py",
        marker="orchestrator.py",
        runner="python",
    ),
}

CONTAINER_WORKDIR = "/testbed"
# Bounds the recorded patch. A model_patch far past this is a runaway write, not
# a fix, and the harness would spend a container slot failing to apply it.
MAX_PATCH_BYTES = 1_000_000

_print_lock = threading.Lock()
_write_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


class StepError(RuntimeError):
    """A step failed in a way that costs this instance but not the run."""


# --------------------------------------------------------------------------
# Subprocess helpers
# --------------------------------------------------------------------------


def run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = 600,
    check: bool = True,
    merge_stderr: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command to completion, killing its whole process group on timeout.

    `subprocess.run(timeout=...)` kills only the direct child, which for
    `docker` or `jac` leaves the real work orphaned and still holding the
    workspace.

    `merge_stderr` folds stderr into stdout at the file-descriptor level rather
    than concatenating them afterwards. That is the only way to keep the two
    interleaved in the order the child wrote them, which matters for anything
    parsed positionally: an eval script traced with `set -x` prints its markers
    to stderr and the test output to stdout, so captured separately there is no
    longer a "between the markers" to slice.
    """
    proc = subprocess.Popen(
        argv,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        out, err = proc.communicate()
        raise StepError(f"timed out after {timeout}s: {' '.join(argv[:4])}")
    done = subprocess.CompletedProcess(argv, proc.returncode, out, err)
    if check and done.returncode != 0:
        raise StepError(
            f"`{' '.join(argv[:6])}` exited {done.returncode}: "
            f"{(err or out or '').strip()[:600]}"
        )
    return done


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
            proc.wait(timeout=5)
            return
        except (OSError, subprocess.TimeoutExpired):
            continue


def git(ws: Path, *args: str, check: bool = True) -> str:
    # -c safe.directory: the workspace was unpacked by `docker cp`, and on some
    # daemons that lands with an ownership git considers dubious.
    done = run(
        ["git", "-C", str(ws), "-c", f"safe.directory={ws}", *args],
        check=check,
        timeout=300,
    )
    return done.stdout


# --------------------------------------------------------------------------
# Docker
# --------------------------------------------------------------------------


def docker_available() -> str:
    if shutil.which("docker") is None:
        return "the `docker` CLI is not on PATH"
    probe = run(["docker", "version", "--format", "{{.Server.Version}}"],
                check=False, timeout=60)
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip().splitlines()
        hint = detail[0] if detail else "unknown error"
        if "permission denied" in hint:
            hint += (
                "\n  Fix with:  sudo usermod -aG docker $USER"
                "\n  then start a new login shell (or run: newgrp docker)."
            )
        return f"the docker daemon is not reachable: {hint}"
    return ""


def ensure_image(image: str, pull_timeout: float) -> None:
    if run(["docker", "image", "inspect", image], check=False, timeout=120).returncode == 0:
        return
    log(f"    pulling {image}")
    run(["docker", "pull", image], timeout=pull_timeout)


def materialize(image: str, ws: Path, timeout: float) -> None:
    """Copy the image's /testbed onto the host at `ws`."""
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    # `true` overrides CMD: this container is never started, it only exists to
    # give `docker cp` a filesystem to read.
    cid = run(["docker", "create", image, "true"], timeout=300).stdout.strip()
    try:
        run(["docker", "cp", f"{cid}:{CONTAINER_WORKDIR}/.", str(ws)], timeout=timeout)
    finally:
        run(["docker", "rm", "-f", cid], check=False, timeout=120)


def start_container(name: str, image: str, ws: Path, network: str) -> None:
    run(["docker", "rm", "-f", name], check=False, timeout=120)
    argv = [
        "docker", "run", "--detach", "--name", name,
        "--volume", f"{ws}:{CONTAINER_WORKDIR}",
        "--workdir", CONTAINER_WORKDIR,
    ]
    if network:
        argv += ["--network", network]
    # Mirrors the official harness's idle container: override CMD, leave the
    # image's entrypoint alone.
    argv += [image, "tail", "-f", "/dev/null"]
    run(argv, timeout=600)


def container_alive(name: str) -> bool:
    probe = run(["docker", "exec", name, "true"], check=False, timeout=120)
    return probe.returncode == 0


def install_commands(eval_script: str) -> list[str]:
    """The setup lines the instance's own eval script runs before testing.

    Read straight from the dataset rather than guessed per-repo. The scripts
    all have the same shape: activation, a few git diagnostics, the install,
    then `git checkout <sha> <test files>` and the test patch. So the boundary
    is the first checkout, and the diagnostics in between are skipped by name --
    including `git -c core.fileMode=false diff <sha>`, which is a diff and not
    a checkout however much it looks like the start of the test setup.
    """
    out: list[str] = []
    for raw in eval_script.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ":", "set ")):
            continue
        if line.startswith(("git checkout", "git apply")):
            break  # the test setup starts here; stop before it
        if line.startswith(("source ", "conda ", "cd ", "git config",
                            "git status", "git show", "git diff", "git -c ")):
            continue
        out.append(line)
    return out


def as_editable(cmd: str) -> str | None:
    """`pip install .` -> `pip install -e .`, or None if it is already fine.

    An editable install points the conda env at /testbed, which the bind mount
    replaces with the agent's workspace -- so an edit takes effect on the next
    import, and nothing needs re-running. A plain `pip install .` instead copies
    the tree into site-packages, and the agent would spend the whole run testing
    against the code it started with. Six of SWE-bench Lite's instances (all of
    psf/requests) install that way.
    """
    toks = cmd.split()
    if "pip" not in toks or "install" not in toks:
        return None
    if "-e" in toks or "--editable" in toks:
        return None
    cut = toks.index("install") + 1
    return " ".join(toks[:cut] + ["-e"] + toks[cut:])


def preparation_script(inst: dict, mode: str) -> list[str]:
    """Which of the instance's install lines this run should actually execute."""
    cmds = install_commands(inst.get("eval_script") or "")
    if mode == "never":
        return []
    if mode == "always":
        return cmds
    out: list[str] = []
    for cmd in cmds:
        toks = cmd.split()
        if "pip" in toks and "install" in toks:
            # Rebuilding an already-editable install costs minutes on the
            # compiled repos and changes nothing, so only fix what is broken.
            editable = as_editable(cmd)
            if editable:
                out.append(editable)
        else:
            out.append(cmd)  # locale-gen and friends: cheap, occasionally load-bearing
    return out


def prepare_workspace(runtime, name: str, inst: dict, mode: str,
                      timeout: float) -> str:
    """Run that install step in the container. Returns "" or why it failed."""
    cmds = preparation_script(inst, mode)
    if not cmds:
        return ""
    # A shell is fine here -- this is harness code running a command out of the
    # dataset, not the agent's screened run_command.
    script = " && ".join(
        ["source /opt/miniconda3/bin/activate", "conda activate testbed", *cmds]
    )
    code, output = runtime.exec_script(name, script, timeout)
    if code != 0:
        tail = output.strip().splitlines()[-3:]
        return f"install step exited {code}: {' / '.join(tail)[:300]}"
    return ""


# --------------------------------------------------------------------------
# Runtimes
#
# Two ways to get an instance image onto this machine and run commands in it.
# `solve_instance` talks only to this interface, so the workspace, the
# objective, the preparation step and the patch extraction are shared and only
# the container mechanics differ.
# --------------------------------------------------------------------------


class DockerRuntime:
    """The daemon. Needs membership of the `docker` group."""

    name = "docker"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def preflight(self) -> str:
        return docker_available()

    def ensure_image(self, image: str) -> None:
        ensure_image(image, self.args.pull_timeout)

    def create(self, name: str, image: str, ws: Path) -> Path:
        # Two steps, because a bind mount cannot show the image's own /testbed:
        # copy it out first, then mount the copy back over the same path.
        materialize(image, ws, self.args.copy_timeout)
        start_container(name, image, ws, self.args.network)
        return ws

    def alive(self, name: str) -> bool:
        return container_alive(name)

    def exec_script(self, name: str, script: str, timeout: float) -> tuple[int, str]:
        done = run(
            ["docker", "exec", "--workdir", CONTAINER_WORKDIR, name,
             "bash", "-c", script],
            check=False, timeout=timeout, merge_stderr=True,
        )
        return done.returncode, (done.stdout or "")

    def reclaim(self, name: str, ws: Path) -> None:
        reclaim(name, ws)

    def destroy(self, name: str, ws: Path, keep_workspace: bool) -> None:
        stop_container(name)
        if not keep_workspace and ws.exists():
            shutil.rmtree(ws, ignore_errors=True)

    def remove_image(self, image: str) -> None:
        run(["docker", "rmi", image], check=False, timeout=600)


class UdockerRuntime:
    """Pure userspace: no daemon, no root, no setuid helpers.

    udocker pulls the image itself and unpacks it into a plain directory, so
    the container's /testbed *is* a host directory. That deletes three steps
    the docker path needs -- the `docker cp` out, the bind mount back, and the
    chown that undoes root-owned leavings -- because there is only ever one
    copy of the tree and it already belongs to this user.

    Commands run under PRoot, which fakes uid 0 by intercepting syscalls rather
    than by holding any privilege.
    """

    name = "udocker"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.bin = args.udocker
        self.env = dict(os.environ)
        self.env["UDOCKER_DIR"] = str(args.udocker_dir)
        # udocker prints a banner and progress to stdout; the driver captures
        # it, so only the exit codes below actually matter.
        self._pulled: set[str] = set()
        self._pull_lock = threading.Lock()

    def _udocker(self, *argv: str, timeout: float = 600,
                 check: bool = True) -> subprocess.CompletedProcess:
        return run([self.bin, *argv], env=self.env, timeout=timeout, check=check)

    def preflight(self) -> str:
        if shutil.which(self.bin) is None:
            return (f"the `{self.bin}` CLI is not on PATH. Install it with:\n"
                    "  pip install udocker && udocker install")
        probe = run([self.bin, "--version"], env=self.env, check=False, timeout=120)
        if probe.returncode != 0:
            return ("udocker is installed but not working: "
                    + (probe.stderr or probe.stdout or "").strip()[:200])
        # `udocker install` fetches the PRoot engines; without them `run` fails
        # per instance instead of once here.
        engines = self.args.udocker_dir / "bin"
        if not any(engines.glob("proot-*")):
            return (f"udocker has no execution engine under {engines}. "
                    "Fetch it with:  udocker install")
        return ""

    def ensure_image(self, image: str) -> None:
        # Serialised: udocker's layer cache is a plain directory with no
        # locking, and two workers pulling the same image race over it. Distinct
        # images still queue behind each other, which costs only the first run.
        with self._pull_lock:
            if image in self._pulled:
                return
            listing = self._udocker("images", check=False, timeout=300)
            if image not in (listing.stdout or ""):
                log(f"    pulling {image}")
                self._udocker("pull", image, timeout=self.args.pull_timeout)
            self._pulled.add(image)

    def container_root(self, name: str) -> Path:
        return self.args.udocker_dir / "containers" / name / "ROOT"

    def create(self, name: str, image: str, ws: Path) -> Path:
        self._udocker("rm", "-f", name, check=False, timeout=300)
        self._udocker("create", f"--name={name}", image, timeout=900)
        root = self.container_root(name)
        testbed = root / CONTAINER_WORKDIR.lstrip("/")
        if not testbed.is_dir():
            raise StepError(
                f"the image unpacked with no {CONTAINER_WORKDIR}: {testbed}"
            )
        # The workspace IS the container, so `ws` (which the docker path would
        # have populated) is ignored and the real path is returned instead.
        return testbed

    def alive(self, name: str) -> bool:
        return self.container_root(name).is_dir()

    def exec_script(self, name: str, script: str, timeout: float) -> tuple[int, str]:
        done = run(
            [self.bin, "run", "--nobanner", f"--workdir={CONTAINER_WORKDIR}",
             name, "bash", "-c", script],
            env=self.env, check=False, timeout=timeout, merge_stderr=True,
        )
        return done.returncode, (done.stdout or "")

    def reclaim(self, name: str, ws: Path) -> None:
        # Nothing to do: PRoot's fake root never left a file this user cannot
        # read, because every write was made with this uid all along.
        return

    def destroy(self, name: str, ws: Path, keep_workspace: bool) -> None:
        if keep_workspace:
            return  # removing the container would remove the workspace with it
        self._udocker("rm", "-f", name, check=False, timeout=600)

    def remove_image(self, image: str) -> None:
        self._udocker("rmi", image, check=False, timeout=600)


RUNTIMES = {"docker": DockerRuntime, "udocker": UdockerRuntime}


def stop_container(name: str) -> None:
    run(["docker", "rm", "--force", "--volumes", name], check=False, timeout=180)


def reclaim(name: str, ws: Path) -> None:
    """Give the workspace back to this user before touching it from the host.

    Commands run in the container as root, so anything they created -- caches,
    build output, a file the agent's own script wrote -- lands root-owned inside
    the bind mount, and the host cannot then delete the workspace.
    """
    run(
        ["docker", "exec", "--user", "0", name,
         "chown", "-R", f"{os.getuid()}:{os.getgid()}", CONTAINER_WORKDIR],
        check=False,
        timeout=600,
    )


# --------------------------------------------------------------------------
# Patch extraction
# --------------------------------------------------------------------------


# Paths a fix never consists of. The baseline comparison already excludes build
# output the image shipped with; this excludes what the run itself leaves behind
# -- caches from the test command, compiled output from a rebuild -- which the
# baseline cannot know about. A repo that gitignores these produces them anyway
# under a different name often enough to be worth naming explicitly.
NOISE_DIRS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache",
    ".hypothesis", ".tox", ".nox", ".eggs", ".jac", ".git", "node_modules",
    "htmlcov", ".benchmarks",
}
NOISE_SUFFIXES = (
    ".pyc", ".pyo", ".pyd", ".so", ".o", ".a", ".dylib", ".dll",
    ".orig", ".rej", ".log", ".coverage",
)
NOISE_TOPLEVEL = {"build", "dist"}


def is_noise(path: str) -> bool:
    parts = path.split("/")
    if parts[0] in NOISE_TOPLEVEL:
        return True
    for part in parts:
        if part in NOISE_DIRS or part.endswith(".egg-info"):
            return True
    return path.endswith(NOISE_SUFFIXES) or parts[-1] == ".coverage"


def porcelain(ws: Path) -> set[tuple[str, str]]:
    """(status, path) for every path git considers dirty or untracked."""
    raw = git(ws, "status", "--porcelain=v1", "-uall", "-z")
    entries: set[tuple[str, str]] = set()
    fields = raw.split("\0")
    i = 0
    while i < len(fields):
        field = fields[i]
        i += 1
        if len(field) < 4:
            continue
        code, path = field[:2], field[3:]
        entries.add((code, path))
        # A rename record is followed by its source path in its own field.
        if "R" in code or "C" in code:
            if i < len(fields):
                entries.add((code, fields[i]))
                i += 1
    return entries


def extract_patch(ws: Path, baseline: set[tuple[str, str]]) -> str:
    """The diff for what this run changed, and nothing else.

    Not simply `git diff`: the tree handed to the agent is the image's /testbed,
    which for several SWE-bench repos already carries untracked build output
    (compiled extensions, .egg-info, generated headers). Diffing against the
    commit would sweep all of that into the prediction and the harness would
    reject the patch. So the baseline is the workspace as it was handed over,
    and only paths whose status changed since are staged.
    """
    after = porcelain(ws)
    after_paths = {p for _, p in after}
    dirty = {p for st, p in after if (st, p) not in baseline}
    # A path that was dirty when we handed the tree over and is clean now was
    # reverted by the run, which is still a change worth diffing.
    reverted = {p for _, p in baseline if p not in after_paths}
    touched = sorted(p for p in (dirty | reverted) if p)
    changed = [p for p in touched if not is_noise(p)]
    dropped = len(touched) - len(changed)
    if dropped:
        log(f"    ignoring {dropped} generated path(s) when building the patch")
    if not changed:
        return ""
    # -A so deletions stage as deletions; -- so a path that looks like a flag
    # cannot become one. Chunked because a run that touches hundreds of files
    # would otherwise build a command line the kernel refuses.
    for i in range(0, len(changed), 200):
        git(ws, "add", "-A", "--", *changed[i : i + 200])
    patch = git(ws, "diff", "--cached", "--no-color", "--no-ext-diff")
    if len(patch.encode("utf-8")) > MAX_PATCH_BYTES:
        raise StepError(
            f"patch is {len(patch.encode('utf-8'))} bytes, over the "
            f"{MAX_PATCH_BYTES} limit ({len(changed)} paths changed)"
        )
    return patch


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------

OBJECTIVE = """\
Resolve the following issue reported against the {repo} repository.

<issue>
{problem_statement}
</issue>

The repository is checked out at commit {base_commit} and is the directory you
are working in. Its dependencies are already installed, and `run_command` runs
inside that prepared environment.

What is expected of you:
  * Find the root cause in the library source and fix it there. A fix that
    special-cases the example in the issue is not a fix.
  * Edit only non-test source files. The tests that grade this work are held
    back and any change you make to a test file is discarded before grading, so
    editing tests can only cost you.
  * Do not change dependencies, build configuration, or version numbers.
  * Verify by running the existing tests that cover the code you touched, with
    `pytest <path>` or whatever runner this repository uses -- for example
    Django's suite runs through `python tests/runtests.py <label>`. Report what
    the output actually said.

Keep the change as small as the issue allows.
"""


def build_objective(inst: dict) -> str:
    return OBJECTIVE.format(
        repo=inst["repo"],
        problem_statement=inst["problem_statement"].strip(),
        base_commit=inst["base_commit"],
    )


def agent_env(args: argparse.Namespace, container: str) -> dict[str, str]:
    env = dict(os.environ)
    env["CODEAGENT_HOME"] = str(args.agent_home)
    env["CODEAGENT_MODEL"] = args.model
    env["CODEAGENT_EXEC"] = args.exec_backend
    env["CODEAGENT_EXEC_CONTAINER"] = container
    if args.exec_backend == "udocker":
        env["CODEAGENT_UDOCKER"] = args.udocker
        env["UDOCKER_DIR"] = str(args.udocker_dir)
    env["CODEAGENT_EXEC_WORKDIR"] = CONTAINER_WORKDIR
    if args.exec_user:
        env["CODEAGENT_EXEC_USER"] = args.exec_user
    env["PYTHONUNBUFFERED"] = "1"
    return env


def agent_argv(args: argparse.Namespace, *extra: str) -> list[str]:
    """How this framework's shim is spawned. The only fork in the driver."""
    if args.fw.runner == "jac":
        return [args.jac, "run", str(args.fw.entry), *extra]
    return [args.python, str(args.fw.entry), *extra]


def run_agent(args: argparse.Namespace, inst: dict, ws: Path,
              container: str, work: Path) -> dict:
    job = work / "job.json"
    result = work / "result.json"
    # A --force re-run reuses this directory, and a stale result file would be
    # read back as if this attempt had produced it.
    result.unlink(missing_ok=True)
    job.write_text(json.dumps({
        "instance_id": inst["instance_id"],
        "objective": build_objective(inst),
        "repo_root": str(ws),
        "max_steps": args.max_steps,
    }, indent=2), encoding="utf-8")

    argv = agent_argv(args, str(job), str(result))
    try:
        done = run(
            argv,
            cwd=args.agent_home,
            env=agent_env(args, container),
            timeout=args.instance_timeout,
            check=False,
        )
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
        "steps": 0, "llm_calls": 0, "prompt_tokens": 0,
        "completion_tokens": 0, "tool_calls": [], "tool_call_count": 0,
        "wall_clock_sec": 0.0,
    }


# --------------------------------------------------------------------------
# One instance, end to end
# --------------------------------------------------------------------------


def solve_instance(args: argparse.Namespace, inst: dict) -> dict:
    iid = inst["instance_id"]
    began = time.time()
    ws = args.workspaces / iid
    work = args.logs / iid
    work.mkdir(parents=True, exist_ok=True)
    # Framework-prefixed: an A/B run has both agents in flight over the same
    # instance ids, and a shared container name would have them fight over one.
    runtime = args.runtime_impl
    container = f"{args.framework}-swe-{iid}-{args.run_id}"[:120]
    record: dict = {
        "instance_id": iid,
        "repo": inst["repo"],
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
            log(f"[{iid}] image HEAD {head[:12]} != base {inst['base_commit'][:12]}"
                f"; resetting")
            git(ws, "checkout", "--force", inst["base_commit"])
        baseline = porcelain(ws)
        prepared = True
        if not runtime.alive(container):
            raise StepError(f"the workspace container {container} did not stay up")
        if preparation_script(inst, args.prepare):
            log(f"[{iid}] wiring the environment to the workspace")
            trouble = prepare_workspace(runtime, container, inst, args.prepare,
                                        args.install_timeout)
            if trouble:
                # Not fatal, but recorded: the failure mode it leaves behind is
                # the agent testing against code it did not write, which looks
                # like a confused agent rather than a broken workspace.
                note(f"workspace preparation: {trouble}")

        log(f"[{iid}] running agent")
        outcome = run_agent(args, inst, ws, container, work)
        if outcome.get("error"):
            note(str(outcome["error"]))
        record.update({
            k: v for k, v in outcome.items() if k not in ("instance_id", "error")
        })
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
            record["patch"] = extract_patch(ws, baseline)
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


def warm_agent(args: argparse.Namespace) -> None:
    """Load the agent once, before the workers can do it simultaneously.

    For byLLM this is a real race: every worker runs `jac run` from the same
    project and the compiled-module cache under .jac/ is shared, so letting N
    of them compile the same modules at once is a fight over the same cache
    files. One warm run makes the rest hits.

    For LangGraph it is a fail-fast check instead -- an unimportable agent or a
    missing dependency surfaces here rather than 300 containers later.

    Either way: invoked with no arguments a shim prints its usage and exits
    non-zero, after importing the whole agent, which is the part being warmed.
    """
    done = run(agent_argv(args), cwd=args.agent_home,
               env=agent_env(args, ""), check=False, timeout=600)
    if "usage:" not in (done.stdout or "") + (done.stderr or ""):
        log("warning: the agent did not load cleanly during warm-up:\n"
            + ((done.stderr or done.stdout or "").strip()[:800] or "(no output)"))


def append(path: Path, row: dict) -> None:
    with _write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def load_instances(args: argparse.Namespace) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    rows = [dict(r) for r in ds]
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
        rows = rows[: args.limit]
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run either coding agent over SWE-bench instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--framework", default="byllm", choices=sorted(FRAMEWORKS),
                   help="which agent to run: the Jac/byLLM one in ../byLLM or "
                        "the LangGraph one in ../langgraph")
    p.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
    p.add_argument("--split", default="test")
    p.add_argument("--instance-ids", nargs="*", default=[],
                   help="run only these instances")
    p.add_argument("--repo", nargs="*", default=[],
                   help="run only instances from these repos, e.g. psf/requests")
    p.add_argument("--limit", type=int, default=0,
                   help="run only the first N instances after filtering")
    p.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"),
                   help="names the output directory and the containers")
    p.add_argument("--output-dir", type=Path, default=BRIDGE_DIR / "results")
    p.add_argument("--model", default=os.environ.get("CODEAGENT_MODEL", "gpt-4o"))
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
                   help="the interpreter to run the langgraph agent with; it "
                        "needs that project's dependencies importable")
    p.add_argument("--runtime", default="docker", choices=sorted(RUNTIMES),
                   help="how to get the instance image and run commands in it. "
                        "'udocker' needs no daemon, no root and no docker group "
                        "-- use it when `docker ps` is permission denied")
    p.add_argument("--udocker", default=os.environ.get("CODEAGENT_UDOCKER", "udocker"),
                   help="the udocker CLI, for --runtime udocker")
    p.add_argument("--udocker-dir", type=Path,
                   default=Path(os.environ.get("UDOCKER_DIR",
                                               Path.home() / ".udocker")),
                   help="udocker's image and container repository")
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
                        "a non-editable install, plus cheap setup like locale-gen; "
                        "'always' replays the whole step, which rebuilds compiled "
                        "repos and costs minutes each")
    p.add_argument("--install-timeout", type=float, default=900,
                   help="ceiling for that install step")
    p.add_argument("--instance-timeout", type=float, default=1800,
                   help="wall-clock ceiling for one agent run")
    p.add_argument("--pull-timeout", type=float, default=3600)
    p.add_argument("--copy-timeout", type=float, default=1800)
    p.add_argument("--keep-workspaces", action="store_true",
                   help="keep each workspace after extracting its patch "
                        "(a full repo copy per instance)")
    p.add_argument("--cleanup-images", action="store_true",
                   help="delete each instance image once done; trades bandwidth "
                        "for disk on long runs")
    p.add_argument("--force", action="store_true",
                   help="re-run instances already present in predictions.jsonl")
    args = p.parse_args(argv)

    args.fw = FRAMEWORKS[args.framework]
    args.udocker_dir = args.udocker_dir.resolve()
    # The agent executes commands through whichever runtime holds the container,
    # unless explicitly overridden. Keeping these in step matters: a `docker
    # exec` into a container that only udocker knows about finds nothing.
    if not args.exec_backend:
        args.exec_backend = args.runtime
    if args.exec_backend not in ("docker", "udocker", "local"):
        raise SystemExit(f"unknown --exec-backend: {args.exec_backend}")
    args.runtime_impl = RUNTIMES[args.runtime](args)
    args.agent_home = (args.agent_home or args.fw.home).resolve()
    args.out = (args.output_dir / args.run_id).resolve()
    args.workspaces = args.out / "workspaces"
    args.logs = args.out / "logs"
    args.predictions = args.out / "predictions.jsonl"
    args.runs = args.out / "runs.jsonl"
    if not args.model_name:
        args.model_name = f"{args.framework}-codeagent/{args.model}"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.fw.entry.exists():
        raise SystemExit(f"missing agent entry: {args.fw.entry}")
    if not (args.agent_home / args.fw.marker).exists():
        raise SystemExit(
            f"no {args.framework} agent at --agent-home {args.agent_home} "
            f"(expected {args.fw.marker})"
        )
    if args.fw.runner == "jac" and shutil.which(args.jac) is None:
        raise SystemExit(f"the jac CLI '{args.jac}' is not on PATH")
    if args.fw.runner == "python" and shutil.which(args.python) is None:
        raise SystemExit(f"the interpreter '{args.python}' is not on PATH")
    # The runtime is needed whatever --exec-backend says: the workspace itself
    # comes out of the instance image.
    why = args.runtime_impl.preflight()
    if why:
        raise SystemExit(
            f"--runtime {args.runtime} is unusable: {why}"
            + ("\n\nNo docker group on this machine? Use --runtime udocker, "
               "which needs neither a daemon nor root."
               if args.runtime == "docker" else "")
        )
    if not any(os.environ.get(k) for k in
               ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_API_KEY")):
        log("warning: no provider API key in the environment; "
            "every instance will fail at the first LLM call")

    instances = load_instances(args)
    for d in (args.out, args.workspaces, args.logs):
        d.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.predictions.exists() and not args.force:
        for line in args.predictions.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["instance_id"])
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
        f"\n  python {BRIDGE_DIR / 'evaluate.py'} --predictions {args.predictions}"
        f" --dataset {args.dataset}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
