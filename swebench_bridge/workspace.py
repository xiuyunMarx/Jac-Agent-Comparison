#!/usr/bin/env python3
"""The tree the agent works in, and the patch that comes back out of it.

Three jobs, none of them framework-specific:

  * **the objective** -- the one piece of text every implementation is given,
    identical down to the whitespace, because it is a prompt and prompts are
    where an A/B silently stops being an A/B;
  * **preparation** -- running only the part of the instance's own install step
    that the bind mount actually breaks;
  * **patch extraction** -- which is deliberately not `git diff`.
"""

from __future__ import annotations

from pathlib import Path

from runtime import StepError, git, log

# Bounds the recorded patch. A model_patch far past this is a runaway write, not
# a fix, and the harness would spend a container slot failing to apply it.
MAX_PATCH_BYTES = 1_000_000


# --------------------------------------------------------------------------
# The objective
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


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------


def install_commands(eval_script: str) -> list[str]:
    """The setup lines the instance's own eval script runs before testing.

    Read straight from the dataset rather than guessed per-repo. The scripts all
    have the same shape: activation, a few git diagnostics, the install, then
    `git checkout <sha> <test files>` and the test patch. So the boundary is the
    first checkout, and the diagnostics in between are skipped by name --
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
    import and nothing needs re-running. A plain `pip install .` instead copies
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
    if mode == "never":
        return []
    cmds = install_commands(inst.get("eval_script") or "")
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


def prepare_workspace(runtime, container: str, inst: dict, mode: str,
                      timeout: float) -> str:
    """Run that install step in the container. Returns "" or why it failed."""
    cmds = preparation_script(inst, mode)
    if not cmds:
        return ""
    # A shell is fine here -- this is harness code running a command out of the
    # dataset, not the agent's screened run_command.
    script = " && ".join(
        ["source /opt/miniconda3/bin/activate", "conda activate testbed", *cmds])
    code, output = runtime.exec_script(container, script, timeout)
    if code != 0:
        tail = output.strip().splitlines()[-3:]
        return f"install step exited {code}: {' / '.join(tail)[:300]}"
    return ""


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
    reject the patch. So the baseline is the workspace as it was handed over, and
    only paths whose status changed since are staged.

    Test-file edits are left in: the harness resets the graded test files itself
    before running, so removing them here would only hide what the agent did.
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
        git(ws, "add", "-A", "--", *changed[i:i + 200])
    patch = git(ws, "diff", "--cached", "--no-color", "--no-ext-diff")
    size = len(patch.encode("utf-8"))
    if size > MAX_PATCH_BYTES:
        raise StepError(
            f"patch is {size} bytes, over the {MAX_PATCH_BYTES} limit "
            f"({len(changed)} paths changed)")
    return patch
