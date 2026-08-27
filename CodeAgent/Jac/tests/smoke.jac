"""端到端 smoke：临时 git 仓库里一个明显的 bug，让 agent 真的走一遍图。
    OLLAMA_API_KEY=... jac run tests/smoke.jac
"""
import os;
import subprocess;
import tempfile;
import from orchestrator { solve }
import from nodes { tool_log }

def make_repo() -> str {
    d = tempfile.mkdtemp(prefix="jacagent-smoke-");
    os.makedirs(os.path.join(d, "calc"));
    os.makedirs(os.path.join(d, "tests"));
    with open(os.path.join(d, "calc", "__init__.py"), "w") as f { f.write(""); }
    with open(os.path.join(d, "calc", "ops.py"), "w") as f {
        f.write("def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n");
    }
    with open(os.path.join(d, "tests", "test_ops.py"), "w") as f {
        f.write("from calc.ops import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n");
    }
    subprocess.run(["git", "init", "-q"], cwd=d, check=True);
    subprocess.run(["git", "add", "-A"], cwd=d, check=True);
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"], cwd=d, check=True);
    return d;
}

with entry {
    if not os.environ.get("OLLAMA_API_KEY") {
        print("set OLLAMA_API_KEY first");
    } else {
        d = make_repo();
        issue = "calc.ops.add(2, 3) returns -1 but the docstring says it returns the sum; expected 5.";
        outcome = solve(issue, d);
        answer = outcome.answer;
        print("=== ledger ===\n" + answer);
        print(f"=== stats === steps={outcome.steps} llm_calls={outcome.llm_calls} prompt={outcome.prompt_tokens} completion={outcome.completion_tokens} cached={outcome.cached_tokens}");
        print("=== tool calls ===");
        for c in tool_log { print(f"  {c['name']:16} ok={c['ok']}  {c['args']}"); }
        diff = subprocess.run(["git", "diff", "--no-color"], cwd=d, capture_output=True, text=True).stdout;
        print("=== patch ===\n" + (diff or "(empty)"));
        print("workspace:", d);
    }
}
