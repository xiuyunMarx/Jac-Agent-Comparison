"""Tool-layer checks, no LLM. `python tests/tool_checks.py` -- the twin of ../Jac/tests/tool_checks.jac."""

import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from nodes import ran_since_edit, repo, tool_log, written  # noqa: E402


def make_repo() -> str:
    d = tempfile.mkdtemp(prefix="pyagent-")
    os.makedirs(os.path.join(d, "pkg"))
    os.makedirs(os.path.join(d, "tests"))
    with open(os.path.join(d, "pkg", "calc.py"), "w") as f:
        f.write("def add(a, b):\n    return a - b\n")
    with open(os.path.join(d, "tests", "test_calc.py"), "w") as f:
        f.write("def test_add():\n    assert 1\n")
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    subprocess.run(["git", "add", "-A"], cwd=d, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"], cwd=d, check=True)
    return d


if __name__ == "__main__":
    d = make_repo()
    repo.base_dir = d
    tool_log.clear()
    assert "pkg/calc.py:2:" in repo.grep("return a - b")
    assert "lines 1-2 of 2" in repo.read_file("pkg/calc.py")
    assert "no def/class" not in repo.outline("pkg/calc.py")
    assert repo.replace_in_file("tests/test_calc.py", "assert 1", "assert 0").startswith("BLOCKED")
    assert not written()
    assert repo.replace_in_file("pkg/calc.py", "return a + b", "x").startswith("Error")
    assert not written()
    assert repo.replace_in_file("pkg/calc.py", "return a - b", "return a + b").startswith("OK")
    assert written()
    assert not ran_since_edit()
    assert "3" in repo.run_snippet("from pkg.calc import add\nprint(add(1, 2))")
    assert ran_since_edit()
    assert "return a + b" in repo.view_diff()
    assert repo.revert_file("pkg/calc.py").startswith("OK")
    assert "return a - b" in repo.read_file("pkg/calc.py")
    assert not ran_since_edit()
    assert repo.run_command("rm -rf /").startswith("Error")
    assert repo.run_command("git reset --hard").startswith("Error")
    assert repo._abs("../../etc/passwd") is None
    repo.cleanup()
    assert not os.path.isdir(os.path.join(d, ".agent_scratch"))
    shutil.rmtree(d, ignore_errors=True)
    print("tool-layer checks passed")
