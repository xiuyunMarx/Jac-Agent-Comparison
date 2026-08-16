"""Drivers that run one chat turn against each system under test.

All drivers are pointed at the token-counting proxy via OPENAI_BASE_URL so
every LLM call is logged there; the driver itself only returns
{agent, response} for the turn.
"""

import importlib.util
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402


class DriverError(Exception):
    pass


class JacServeDriver:
    """Runs a Jac project via `jac start main.jac -p <port>` and talks REST."""

    def __init__(self, name: str, project_dir: Path, port: int, log_dir: Path):
        self.name = name
        self.project_dir = project_dir
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.log_path = log_dir / f"{name}_server.log"
        self.proc: subprocess.Popen | None = None

    def start(self, timeout: float = 600.0) -> None:
        common.setup_env()
        env = dict(os.environ)
        env["OPENAI_BASE_URL"] = common.PROXY_BASE_URL
        env["PATH"] = common.JAC_BIN_DIR + os.pathsep + env.get("PATH", "")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("w")
        self.proc = subprocess.Popen(
            ["jac", "start", "main.jac", "-p", str(self.port)],
            cwd=self.project_dir, env=env,
            stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # `jac start` silently falls back to another port if ours is taken, so
        # the log — not the requested port — is the authority on where OUR
        # server actually listens. Never probe a port the log hasn't confirmed.
        deadline = time.time() + timeout
        bound_port = None
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise DriverError(
                    f"{self.name}: server exited early, see {self.log_path}")
            if bound_port is None:
                m = re.search(r"Server running on http://[^:]+:(\d+)",
                              self.log_path.read_text(errors="replace"))
                if m:
                    bound_port = int(m.group(1))
                    if bound_port != self.port:
                        print(f"  {self.name}: port {self.port} was taken, "
                              f"server bound {bound_port}; following it")
                        self.port = bound_port
                        self.base = f"http://127.0.0.1:{bound_port}"
            if bound_port is not None:
                try:
                    r = requests.post(f"{self.base}/walker/get_session",
                                      json={"session_id": "__probe__"}, timeout=5)
                    if r.status_code == 200:
                        return
                except requests.RequestException:
                    pass
            time.sleep(2)
        raise DriverError(f"{self.name}: server not ready after {timeout}s")

    def interact(self, message: str, session_id: str, timeout: float = 600.0) -> dict:
        r = requests.post(f"{self.base}/walker/interact",
                          json={"message": message, "session_id": session_id},
                          timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        reports = (payload.get("data") or {}).get("reports") or payload.get("reports") or []
        if not reports:
            raise DriverError(f"{self.name}: no reports in response: {str(payload)[:500]}")
        rep = reports[0]
        return {"agent": rep.get("agent", ""), "response": rep.get("response", "")}

    def stop(self) -> None:
        if self.proc:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(self.proc.pid), sig)
                except ProcessLookupError:
                    break
                try:
                    self.proc.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    continue
        # The dev server may leave a detached worker holding the port; verify.
        for _ in range(10):
            try:
                requests.get(f"{self.base}/", timeout=1)
                time.sleep(1)
            except requests.RequestException:
                self.proc = None
                return
        print(f"  WARNING: {self.name}: something still answers on {self.base} "
              "after teardown — kill it before the next run")
        self.proc = None


class LanggraphDriver:
    """Runs the LangGraph port in-process (imports jac-gpt.py via importlib)."""

    def __init__(self, name: str, project_dir: Path):
        self.name = name
        self.project_dir = project_dir
        self.factory = None

    def start(self, timeout: float = 600.0) -> None:
        common.setup_env()
        os.environ["OPENAI_BASE_URL"] = common.PROXY_BASE_URL
        os.chdir(self.project_dir)  # its config/docs/faiss paths are cwd-relative
        sys.path.insert(0, str(self.project_dir))
        spec = importlib.util.spec_from_file_location(
            "jac_gpt_langgraph", self.project_dir / "jac-gpt.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Its nested-JSON config parse always falls back to dataclass defaults,
        # which equal config/faiss_reranking.json's values; default path is fine.
        self.factory = module.JacGPTFactory()

    def interact(self, message: str, session_id: str, timeout: float = 600.0) -> dict:
        result = self.factory.interact(message=message, session_id=session_id)
        return {"agent": result.get("agent", ""), "response": result.get("response", "")}

    def stop(self) -> None:
        self.factory = None


def make_driver(system: str, log_dir: Path):
    spec = common.SYSTEMS[system]
    if spec["kind"] == "jac":
        return JacServeDriver(system, spec["dir"], spec["port"], log_dir)
    return LanggraphDriver(system, spec["dir"])
