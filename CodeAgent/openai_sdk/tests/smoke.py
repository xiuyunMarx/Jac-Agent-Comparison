"""End-to-end smoke: an obvious bug in a temporary git repo, the agent walks the whole graph.
    OLLAMA_API_KEY=... python tests/smoke.py
"""

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from nodes import tool_log  # noqa: E402
from orchestrator import solve  # noqa: E402


def make_repo() -> str:
    d = tempfile.mkdtemp(prefix="pyagent-smoke-")
    os.makedirs(os.path.join(d, "calc"))
    os.makedirs(os.path.join(d, "tests"))
    with open(os.path.join(d, "calc", "__init__.py"), "w") as f:
        f.write("")
    with open(os.path.join(d, "calc", "ops.py"), "w") as f:
        f.write("def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n")
    with open(os.path.join(d, "tests", "test_ops.py"), "w") as f:
        f.write("from calc.ops import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n")
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    subprocess.run(["git", "add", "-A"], cwd=d, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"], cwd=d, check=True)
    return d


if __name__ == "__main__":
    if not (os.environ.get("OLLAMA_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("set OLLAMA_API_KEY first")
    else:
        d = make_repo()
        issue = "calc.ops.add(2, 3) returns -1 but the docstring says it returns the sum; expected 5."
        outcome = solve(issue, d)
        print("=== ledger ===\n" + outcome.answer)
        print(f"=== stats === steps={outcome.steps} llm_calls={outcome.llm_calls} "
              f"prompt={outcome.prompt_tokens} completion={outcome.completion_tokens} cached={outcome.cached_tokens}")
        print("=== tool calls ===")
        for c in tool_log:
            print(f"  {c['name']:16} ok={c['ok']}  {c['args']}")
        diff = subprocess.run(["git", "diff", "--no-color"], cwd=d, capture_output=True, text=True).stdout
        print("=== patch ===\n" + (diff or "(empty)"))
        print("workspace:", d)
