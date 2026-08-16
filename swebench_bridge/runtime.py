#!/usr/bin/env python3
"""Getting an instance image onto this machine, and running commands in it.

Two backends behind one interface. `run_agent.py` and `grade.py` talk only to
that interface, so the workspace, the objective, the preparation step, the patch
extraction and the verdict are shared code and only the container mechanics
differ.

  docker   the daemon. Needs membership of the `docker` group.
  udocker  pure userspace: no daemon, no root, no setuid helpers.

udocker is not a second-class path. Same images, same /testbed, same conda env,
same tests. In one respect it is simpler: under docker the image's /testbed has
to be copied out with `docker cp`, bind-mounted back over itself, and chowned
afterwards because everything the container wrote landed root-owned. Under
udocker the container's /testbed already *is* a host directory owned by you, so
all three steps disappear and there is only ever one copy of the tree.

This module also owns the process primitives the rest of the bridge uses --
`run`, `git`, `log`, `StepError` -- because they exist to serve the container
work and putting them anywhere else would make every module import the driver.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

CONTAINER_WORKDIR = "/testbed"

_print_lock = threading.Lock()


class StepError(RuntimeError):
    """One step of one instance failed. Never ends the run."""


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


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


def free_gb(path: Path | None = None) -> float:
    return shutil.disk_usage(path or Path.home()).free / 2**30


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class RuntimeConfig:
    """What a runtime needs, independent of which script is driving it.

    A plain namespace was what the two backends used to read, which coupled them
    to one script's argparse. Grading needs the same backends and has a different
    flag set, so the shared subset is spelled out here instead.
    """

    udocker: str = "udocker"
    udocker_dir: Path = field(default_factory=lambda: Path.home() / ".udocker")
    network: str = ""
    pull_timeout: float = 3600
    copy_timeout: float = 1800


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


class DockerRuntime:
    """The daemon. Needs membership of the `docker` group."""

    name = "docker"

    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg

    def preflight(self) -> str:
        return docker_available()

    def ensure_image(self, image: str) -> None:
        if run(["docker", "image", "inspect", image],
               check=False, timeout=120).returncode == 0:
            return
        log(f"    pulling {image}")
        run(["docker", "pull", image], timeout=self.cfg.pull_timeout)

    def create(self, name: str, image: str, ws: Path) -> Path:
        # Two steps, because a bind mount cannot show the image's own /testbed:
        # copy it out first, then mount the copy back over the same path.
        self._materialize(image, ws)
        self._start(name, image, ws)
        return ws

    def _materialize(self, image: str, ws: Path) -> None:
        """Copy the image's /testbed onto the host at `ws`."""
        if ws.exists():
            shutil.rmtree(ws)
        ws.mkdir(parents=True)
        # `true` overrides CMD: this container is never started, it only exists
        # to give `docker cp` a filesystem to read.
        cid = run(["docker", "create", image, "true"], timeout=300).stdout.strip()
        try:
            run(["docker", "cp", f"{cid}:{CONTAINER_WORKDIR}/.", str(ws)],
                timeout=self.cfg.copy_timeout)
        finally:
            run(["docker", "rm", "-f", cid], check=False, timeout=120)

    def _start(self, name: str, image: str, ws: Path) -> None:
        run(["docker", "rm", "-f", name], check=False, timeout=120)
        argv = [
            "docker", "run", "--detach", "--name", name,
            "--volume", f"{ws}:{CONTAINER_WORKDIR}",
            "--workdir", CONTAINER_WORKDIR,
        ]
        if self.cfg.network:
            argv += ["--network", self.cfg.network]
        # Mirrors the official harness's idle container: override CMD, leave the
        # image's entrypoint alone.
        argv += [image, "tail", "-f", "/dev/null"]
        run(argv, timeout=600)

    def alive(self, name: str) -> bool:
        return run(["docker", "exec", name, "true"],
                   check=False, timeout=120).returncode == 0

    def exec_script(self, name: str, script: str, timeout: float) -> tuple[int, str]:
        done = run(
            ["docker", "exec", "--workdir", CONTAINER_WORKDIR, name,
             "bash", "-c", script],
            check=False, timeout=timeout, merge_stderr=True,
        )
        return done.returncode, (done.stdout or "")

    def reclaim(self, name: str, ws: Path) -> None:
        """Give the workspace back to this user before the host touches it.

        Commands run in the container as root, so anything they created --
        caches, build output, a file the agent's own script wrote -- lands
        root-owned inside the bind mount, and the host cannot then delete it.
        """
        run(
            ["docker", "exec", "--user", "0", name,
             "chown", "-R", f"{os.getuid()}:{os.getgid()}", CONTAINER_WORKDIR],
            check=False, timeout=600,
        )

    def destroy(self, name: str, ws: Path, keep_workspace: bool) -> None:
        run(["docker", "rm", "--force", "--volumes", name],
            check=False, timeout=180)
        if not keep_workspace and ws.exists():
            shutil.rmtree(ws, ignore_errors=True)

    def remove_image(self, image: str) -> None:
        run(["docker", "rmi", image], check=False, timeout=600)


# --------------------------------------------------------------------------
# udocker
# --------------------------------------------------------------------------


class UdockerRuntime:
    """Pure userspace: no daemon, no root, no setuid helpers.

    udocker pulls the image itself and unpacks it into a plain directory, so the
    container's /testbed *is* a host directory. That deletes three steps the
    docker path needs -- the `docker cp` out, the bind mount back, and the chown
    that undoes root-owned leavings -- because there is only ever one copy of the
    tree and it already belongs to this user.

    Commands run under PRoot, which fakes uid 0 by intercepting syscalls rather
    than by holding any privilege.
    """

    name = "udocker"

    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        self.bin = cfg.udocker
        self.env = dict(os.environ)
        self.env["UDOCKER_DIR"] = str(cfg.udocker_dir)
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
        engines = self.cfg.udocker_dir / "bin"
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
                self._udocker("pull", image, timeout=self.cfg.pull_timeout)
            self._pulled.add(image)

    def container_root(self, name: str) -> Path:
        return self.cfg.udocker_dir / "containers" / name / "ROOT"

    def create(self, name: str, image: str, ws: Path) -> Path:
        self._udocker("rm", "-f", name, check=False, timeout=300)
        self._udocker("create", f"--name={name}", image, timeout=900)
        testbed = self.container_root(name) / CONTAINER_WORKDIR.lstrip("/")
        if not testbed.is_dir():
            raise StepError(
                f"the image unpacked with no {CONTAINER_WORKDIR}: {testbed}")
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


def build_runtime(name: str, cfg: RuntimeConfig):
    if name not in RUNTIMES:
        raise SystemExit(f"unknown runtime: {name}")
    return RUNTIMES[name](cfg)


# --------------------------------------------------------------------------
# Disk
# --------------------------------------------------------------------------


class DiskGate:
    """Block a worker until the filesystem can hold another container.

    An instance image is 1-2 GB and a container unpacks to about the same again.
    Running N of them without a floor is how a long run dies two thirds of the
    way through with `No space left on device` -- and under udocker that failure
    lands mid-unpack, leaving a half-written container the next attempt trips
    over. Waiting is always cheaper than that.
    """

    def __init__(self, min_free_gb: float = 6.0, poll_sec: float = 20.0) -> None:
        self.min_free_gb = min_free_gb
        self.poll_sec = poll_sec

    def wait(self, iid: str) -> None:
        warned = False
        while free_gb() < self.min_free_gb:
            if not warned:
                log(f"[{iid}] waiting for room "
                    f"({free_gb():.1f}G free, need {self.min_free_gb}G)")
                warned = True
            time.sleep(self.poll_sec)
