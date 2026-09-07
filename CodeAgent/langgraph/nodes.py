"""Repository tools, shared run state, and phase definitions."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from fnmatch import fnmatch
from functools import cache
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI

MODEL_NAME = os.environ.get("CODEAGENT_MODEL", "openai/glm-5.2")
MODEL_BASE = (os.environ.get("OLLAMA_API_BASE") or os.environ.get("OPENAI_API_BASE")
              or os.environ.get("OPENAI_BASE_URL") or "https://ollama.com/v1")
API_KEY = os.environ.get("OLLAMA_API_KEY") or os.environ.get("OPENAI_API_KEY") or "ollama"
TEMPERATURE = float(os.environ.get("CODEAGENT_TEMPERATURE", "0.7"))


@cache
def get_model() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL_NAME.removeprefix("openai/"), base_url=MODEL_BASE,
                      api_key=API_KEY, temperature=TEMPERATURE)


history: list[BaseMessage] = []         # appended in place, one turn per round
tool_log: list[dict[str, Any]] = []     # one {"name", "ok", "args"} per tool call
TOOL_BUDGET: int = 60                   # budgeted in tool calls, not phases
SCRATCH: str = ".agent_scratch"         # reproduction scripts; removed at Finish
WRITE_TOOLS: list[str] = ["replace_in_file", "write_file"]


def log_call(name: str, ok: bool = True, args: str = "") -> dict[str, Any]:
    call = {"name": name, "ok": ok, "args": args}
    tool_log.append(call)
    return call


def written() -> bool:
    """Whether a successful write really landed -- asks the tool log, not the model."""
    return any(c["name"] in WRITE_TOOLS and c["ok"] for c in tool_log)


def ran_since_edit() -> bool:
    """Whether anything ran after the last write; if not, every test output seen is stale."""
    ran = False
    for c in tool_log:
        if (c["name"] in WRITE_TOOLS and c["ok"]) or c["name"] == "revert_file":
            ran = False
        elif c["name"] in ("run_command", "run_snippet"):
            ran = True
    return ran


class Repo:
    def __init__(self, base_dir: str = ".") -> None:
        self.base_dir = base_dir

    def _abs(self, path: str) -> str | None:
        base = os.path.realpath(self.base_dir)
        cand = os.path.realpath(os.path.join(base, path))
        if cand == base or cand.startswith(base + os.sep):
            return cand
        return None

    def _is_test_path(self, path: str) -> bool:
        parts = path.replace("\\", "/").split("/")
        name = parts[-1]
        return ("tests" in parts) or ("test" in parts) or name.startswith("test_") or name.endswith("_test.py")

    def read_file(self, path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a file from the repository. Files up to 400 lines come back whole; larger ones come back as the requested window. Content has no line-number prefixes, so it can be copied verbatim as an edit anchor.

        Args:
            path: Path relative to the repository root.
            start_line: First line to show (1-based).
            end_line: Last line to show, inclusive; 0 means to the end.
        """
        log_call("read_file", args=f"{path}:{start_line}-{end_line}")
        target = self._abs(path)
        if target is None or not os.path.isfile(target):
            return f"Error: no such file '{path}'. Use grep to locate it."
        try:
            with open(target, "r", encoding="utf-8", errors="strict", newline="") as f:
                lines = f.read().splitlines()
        except UnicodeDecodeError:
            return f"Error: '{path}' is not UTF-8 text."
        total = len(lines)
        first = max(1, start_line)
        last = total if end_line <= 0 else min(total, end_line)
        if total <= 400:            # small files come back whole; no windowing round trips
            first, last = 1, total
        body = "\n".join(lines[first - 1:last])
        return f"# {path} lines {first}-{last} of {total}\n{body}"

    def grep(self, pattern: str, path: str = ".", file_glob: str = "*.py") -> str:
        """Search file contents with a Python regex. Returns 'path:line: text' per match. Use this to find where a symbol is defined or used.

        Args:
            pattern: Python regular expression.
            path: File or directory to search, relative to the repository root.
            file_glob: Glob restricting file names, e.g. '*.py'.
        """
        log_call("grep", args=f"{pattern} in {path} ({file_glob})")
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"Error: bad regex '{pattern}': {e}"
        base = self._abs(path)
        if base is None:
            return f"Error: '{path}' is outside the repository."
        hits: list[str] = []
        for d, dirs, files in os.walk(base):
            dirs[:] = sorted(x for x in dirs if not x.startswith(".") and x not in ("build", "dist", "__pycache__", "node_modules"))
            for name in sorted(files):
                if not fnmatch(name, file_glob):
                    continue
                full = os.path.join(d, name)
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except OSError:
                    continue
                rel = os.path.relpath(full, os.path.realpath(self.base_dir))
                for i, line in enumerate(text.splitlines()):
                    if rx.search(line):
                        hits.append(f"{rel}:{i + 1}: {line.strip()[:200]}")
                    if len(hits) >= 100:
                        return "\n".join(hits) + "\n... [100 matches; narrow the pattern]"
        return "\n".join(hits) if hits else f"No matches for '{pattern}' under '{path}'."

    def outline(self, path: str) -> str:
        """List every def/class in a file with its line number. Much cheaper than reading the whole file; use it to decide which window to read.

        Args:
            path: Path relative to the repository root.
        """
        log_call("outline", args=path)
        target = self._abs(path)
        if target is None or not os.path.isfile(target):
            return f"Error: no such file '{path}'."
        with open(target, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
        out = [f"{i + 1}: {line.rstrip()}" for i, line in enumerate(lines)
               if re.match(r"^\s*(async\s+def|def|class)\s+\w+", line)]
        return "\n".join(out) if out else f"({path}: no def/class found)"

    def replace_in_file(self, path: str, old: str, new: str) -> str:
        """Replace an exact, unique piece of text in a file. Literal match, not regex. Test files are refused. If 'old' is not found or not unique, the file is unchanged and the error says what to do.

        Args:
            path: Path relative to the repository root.
            old: Exact existing text, copied character for character including indentation.
            new: Replacement text.
        """
        call = log_call("replace_in_file", ok=False, args=path)
        target = self._abs(path)
        if target is None or not os.path.isfile(target):
            return f"Error: no such file '{path}'."
        if self._is_test_path(path):
            return "BLOCKED: test files are reset before grading; edit the library source instead."
        if not old:
            return "Error: 'old' must not be empty."
        with open(target, "r", encoding="utf-8", newline="") as f:
            text = f.read()
        n = text.count(old)
        if n == 0:
            hint = " A near-match differs only in whitespace; re-read the file and copy exactly." if old.strip() in text else ""
            return f"Error: 'old' not found in {path}.{hint}"
        if n > 1:
            return f"Error: 'old' occurs {n} times in {path}; add surrounding lines to make it unique."
        updated = text.replace(old, new)
        with open(target, "w", encoding="utf-8", newline="") as f:
            f.write(updated)
        call["ok"] = True
        warn = ""
        if path.endswith(".py"):
            try:
                compile(updated, path, "exec")
            except SyntaxError as e:
                warn = f"\nWARNING: {path} no longer parses at line {e.lineno}: {e.msg}. Fix before running tests."
        line_no = text[:text.index(old)].count("\n") + 1
        return f"OK: replaced 1 occurrence in {path} at line {line_no}.{warn}"

    def view_diff(self) -> str:
        """Show the git diff of everything changed so far in this run.
        """
        log_call("view_diff")
        return self._exec(["git", "diff", "--no-color", "--", ".", f":(exclude){SCRATCH}"])

    def run_command(self, command: str) -> str:
        """Run one command in the prepared environment, with the working directory already at the repository root: pytest <path>, python <script or -c ...>, or read-only git (status/diff/log/show). This is not a shell: no cd, pipes, redirects, && or environment-variable prefixes -- pass a single command line and the full output comes back with the exit code.

        Args:
            command: The command line, e.g. 'pytest tests/test_x.py -k name'.
        """
        log_call("run_command", args=command)
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return f"Error: cannot parse command: {e}"
        if not argv:
            return "Error: empty command."
        allowed = ["pytest", "python", "python3", "git"]
        if argv[0] not in allowed:
            return f"Error: '{argv[0]}' is not allowed. Allowed: {', '.join(allowed)}."
        if argv[0] == "git" and (len(argv) < 2 or argv[1] not in ("status", "diff", "log", "show")):
            return "Error: only read-only git subcommands (status/diff/log/show) are allowed."
        if argv[0] == "pytest":
            argv = argv + ["-q", "--tb=line", "-p", "no:cacheprovider"]
        return self._exec(argv)

    def run_snippet(self, code: str) -> str:
        """Run a short Python script (for example a reproduction of the issue) inside the prepared environment. Returns exit code and output.

        Args:
            code: Complete Python source of the script.
        """
        log_call("run_snippet", args=code[:80])
        scratch = os.path.join(os.path.realpath(self.base_dir), SCRATCH)
        os.makedirs(scratch, exist_ok=True)
        with open(os.path.join(scratch, "snippet.py"), "w", encoding="utf-8") as f:
            f.write(code)
        # `python -c` keeps sys.path[0] at the cwd (the repo root), not the scratch dir.
        return self._exec([
            "python", "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__main__')",
            f"{SCRATCH}/snippet.py",
        ])

    # Inside the container, activate the image's conda env first (the SWE-bench
    # deps live in `testbed`; a bare `docker exec ... python` hits the system
    # python), then exec the original command; "$@" keeps argv unre-parsed.
    def _in_env(self, argv: list[str]) -> list[str]:
        return ["bash", "-c",
                "source /opt/miniconda3/bin/activate >/dev/null 2>&1; "
                "conda activate testbed >/dev/null 2>&1; exec \"$@\"", "_"] + argv

    # The single execution point. CODEAGENT_EXEC=docker|udocker runs in the container.
    def _exec(self, argv: list[str]) -> str:
        backend = os.environ.get("CODEAGENT_EXEC", "local")
        workdir = os.environ.get("CODEAGENT_EXEC_WORKDIR", "/testbed")
        container = os.environ.get("CODEAGENT_EXEC_CONTAINER", "")
        if backend == "docker":
            prefix = ["docker", "exec", "-w", workdir]
            user = os.environ.get("CODEAGENT_EXEC_USER", "")
            if user:
                prefix += ["-u", user]
            argv = prefix + [container] + self._in_env(argv)
        elif backend == "udocker":
            argv = [os.environ.get("CODEAGENT_UDOCKER", "udocker"), "run", "--nobanner",
                    f"--workdir={workdir}", container] + self._in_env(argv)
        try:
            done = subprocess.run(argv, cwd=self.base_dir, capture_output=True, text=True,
                                  timeout=600, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 600s."
        except OSError as e:
            return f"Error: could not run command: {e}"
        out = (done.stdout or "") + (done.stderr or "")
        if len(out) > 12000:
            out = out[:6000] + "\n... [omitted] ...\n" + out[-6000:]
        return f"exit code {done.returncode}\n{out}" if out else f"exit code {done.returncode} (no output)"

    def revert_file(self, path: str) -> str:
        """Undo every change this run made to one file, restoring the original. Use it when an edit turned out to be in the wrong place.

        Args:
            path: Path relative to the repository root.
        """
        log_call("revert_file", args=path)
        if self._abs(path) is None:
            return f"Error: '{path}' is outside the repository."
        out = self._exec(["git", "checkout", "--", path])
        return f"OK: reverted {path} to its original contents." if out.startswith("exit code 0") else out

    def cleanup(self) -> None:
        scratch = os.path.join(os.path.realpath(self.base_dir), SCRATCH)
        if os.path.isdir(scratch):
            shutil.rmtree(scratch, ignore_errors=True)



repo = Repo()


@dataclass(frozen=True)
class Phase:
    name: str
    goal: str
    tools: tuple[str, ...]
    max_react_iterations: int
    sem: str

    def toolbox(self) -> list[BaseTool]:
        return [StructuredTool.from_function(getattr(repo, name), parse_docstring=True)
                for name in self.tools]


PHASES: dict[str, Phase] = {
    "Plan": Phase(
        name="Plan",
        goal="Restate the issue as one falsifiable claim (what call misbehaves, what it should do). Use grep/outline to name the 1-3 source files most likely involved. Do not edit.",
        tools=("grep", "outline", "read_file"),
        max_react_iterations=6,
        sem="Carry out the planning phase with the tools provided, then reply with a short summary the next phase needs."),
    "Explore": Phase(
        name="Explore",
        goal="Locate the exact function and lines to change. Write a minimal reproduction with run_snippet and confirm it currently fails. Report file, symbol, line range, and the repro output.",
        tools=("grep", "outline", "read_file", "run_snippet", "view_diff", "revert_file"),
        max_react_iterations=20,
        sem="Carry out the locating phase with the tools provided, then reply with a short summary the next phase needs."),
    "Edit": Phase(
        name="Edit",
        goal="Apply the smallest fix in library source with replace_in_file (actually call it -- describing an edit is not an edit). Re-run the reproduction with run_snippet; it must now pass. Check view_diff.",
        tools=("read_file", "grep", "outline", "replace_in_file", "run_snippet", "view_diff"),
        max_react_iterations=20,
        sem="Carry out the editing phase with the tools provided, then reply with a short summary the next phase needs."),
    "Verify": Phase(
        name="Verify",
        goal="Run the repository's existing tests nearest the changed code (pytest <path>, or the repo's own runner). Report exactly what the output said: passed/failed counts and any failure lines.",
        tools=("run_command", "view_diff", "read_file"),
        max_react_iterations=10,
        sem="Carry out the verification phase with the tools provided, then reply with a short summary of the test results."),
    "Finish": Phase(name="Finish", goal="Done.", tools=(), max_react_iterations=0, sem=""),
}


@dataclass(frozen=True)
class PhaseCapability:
    """The edge between phases. `capability` is the router's hint: what the phase it leads to can do."""

    target: str
    capability: str
    kind: str = "forward"       # forward | repair | relocate | finish


EDGES: dict[str, tuple[PhaseCapability, ...]] = {
    "Plan": (PhaseCapability("Explore", "locate the exact code to change and reproduce the failure"),),
    "Explore": (PhaseCapability("Edit", "apply the fix to library source and re-run the reproduction"),),
    "Edit": (PhaseCapability("Verify", "run the repository's own tests against the edited tree"),),
    "Verify": (
        PhaseCapability("Edit", "repair: the fix is in the right place but incomplete or wrong -- edit it again", kind="repair"),
        PhaseCapability("Explore", "relocate: the reproduction still fails exactly as before, so the edit is in the wrong place -- revert it and find the real root cause", kind="relocate"),
        PhaseCapability("Finish", "finish: the reproduction and the covering tests pass after the last edit", kind="finish"),
    ),
    "Finish": (),
}


def edges_from(phase: str, target: str | None = None, exclude_kind: str | None = None) -> list[PhaseCapability]:
    """Outgoing transitions allowed for this decision."""
    out = [e for e in EDGES[phase] if exclude_kind is None or e.kind != exclude_kind]
    if target is not None:
        out = [e for e in out if e.target == target]
    return out
