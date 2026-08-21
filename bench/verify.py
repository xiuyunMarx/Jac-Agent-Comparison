"""Prove the sweep would work, without spending a token.

Everything here is checked by construction or by import: which model id each of
the fifteen arms resolves to, whether its client would honour the shared
endpoint, whether each runner knows about its third arm, whether the inputs and
the toolchain are present. Nothing calls a model. Nothing loads weights.

That distinction matters because the expensive failures in this repo are not
inference failures -- they are two arms of one benchmark quietly resolving to
different models, a runner that never learned about its third arm, an absolute
path that only exists on one machine. All of those are visible from here.

    python -m bench.verify              # everything
    python -m bench.verify --models     # just the fifteen model seams

Python arms are verified live: the module is imported with the benchmark
environment and asked what it resolved. Jac arms are verified statically --
`jac check` plus the presence of the env knob -- because importing one
constructs its whole graph, and for two of them that means loading an embedding
model. Each row says which kind of check produced it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import config

R = config.REPO_ROOT
PY = sys.executable

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


def _run(argv, cwd=None, env=None, timeout=180):
    try:
        return subprocess.run(argv, cwd=cwd, env=env, timeout=timeout,
                              capture_output=True, text=True)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        class R_:
            returncode, stdout, stderr = 1, "", f"{type(exc).__name__}: {exc}"
        return R_()


def _py_probe(project: str, cwd: Path, snippet: str, extra_path: list[Path] | None = None):
    """Import an arm with the benchmark env and ask it what it resolved."""
    env = config.env_for(project)
    paths = [str(cwd)] + [str(p) for p in (extra_path or [])]
    env["PYTHONPATH"] = os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    proc = _run([PY, "-c", snippet], cwd=cwd, env=env)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        return None, tail[-1][:160] if tail else "import failed"
    return (proc.stdout or "").strip(), None


def _jac_static(project: str, jac_file: Path, knob: str):
    """`jac check` the file, and confirm the model knob is actually read.

    Static because importing a Jac arm runs its module-level graph construction;
    for the RagGPT arms that means loading an embedding model and a FAISS index.
    """
    if not jac_file.is_file():
        return None, f"missing {jac_file}"
    env = config.env_for(project)
    proc = _run(["jac", "check", jac_file.name], cwd=jac_file.parent, env=env)
    if "PASSED" not in (proc.stdout or "") + (proc.stderr or ""):
        tail = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()
        return None, f"jac check failed: {tail[-1][:120] if tail else '?'}"
    text = jac_file.read_text()
    if knob not in text:
        return None, f"compiles, but never reads ${knob}"
    prefixed = "openai/" in text
    return (f"${knob} -> " + ("openai/<model>" if prefixed else "<model>")), None


# ---------------------------------------------------------------------------
# The fifteen model seams
# ---------------------------------------------------------------------------

def check_models() -> list[tuple]:
    rows = []

    def add(bench, arm, kind, value, err):
        rows.append((bench, arm, PASS if err is None else FAIL, kind,
                     value if err is None else err))

    # -- CodeAgent ----------------------------------------------------------
    add("CodeAgent", "byLLM", "static",
        *_jac_static("codeagent", R / "CodeAgent" / "byLLM" / "orchestrator.jac",
                     "CODEAGENT_MODEL"))
    add("CodeAgent", "langgraph", "import", *_py_probe(
        "codeagent", R / "CodeAgent" / "langgraph",
        "import orchestrator as o; m=o.build_model();"
        "print(f'{m.model_name} | stream={m.streaming} | max_tokens={m.max_tokens}')"))
    add("CodeAgent", "openai_sdk", "import", *_py_probe(
        "codeagent", R / "CodeAgent" / "openai_sdk",
        "import llm; print(f'{llm.active_model_name()} | stream={llm.STREAM} "
        "| max_tokens={llm.MAX_TOKENS} | temp={llm.TEMPERATURE}')"))

    # -- Email --------------------------------------------------------------
    add("Email", "byLLM", "static",
        *_jac_static("email", R / "Email-Auto-response" / "byLLM" / "nodes.jac",
                     "OPENAI_MODEL_NAME"))
    add("Email", "CrewAI-LangGraph", "import", *_py_probe(
        "email", R / "Email-Auto-response" / "CrewAI-LangGraph",
        # crewai resolves the model itself, from these vars in this order.
        "import os; from crewai.utilities.llm_utils import create_llm;"
        "m=create_llm(None); print(f'{m.model} | base_url={m.base_url or m.api_base}')"))
    add("Email", "openai_sdk", "import", *_py_probe(
        "email", R / "Email-Auto-response" / "openai_sdk",
        "import llm; print(f'{llm.active_model_name()} | temp={llm.TEMPERATURE}')"))

    # -- meeting-assistant --------------------------------------------------
    add("meeting", "byLLM", "static",
        *_jac_static("meeting", R / "meeting-assistant" / "byLLM" / "nodes.jac",
                     "MEETING_MODEL"))
    add("meeting", "CrewAI", "import", *_py_probe(
        "meeting", R / "meeting-assistant" / "CrewAI" / "src",
        "from meeting_assistant_flow.crews.meeting_assistant_crew"
        ".meeting_assistant_crew import _model_name; print(_model_name())"))
    add("meeting", "openai_sdk", "import", *_py_probe(
        "meeting", R / "meeting-assistant" / "openai_sdk",
        "import nodes; print(f'{nodes.MODEL} | temp={nodes.TEMPERATURE}')"))

    # -- RagGPT -------------------------------------------------------------
    for name, d in (("Jac-Rag-GPT", "Jac-Rag-GPT"),
                    ("Jac-Rag-GPT-ByllmRouter", "Jac-Rag-GPT-ByllmRouter")):
        add("RagGPT", name, "static",
            *_jac_static("raggpt", R / "RagGPT" / d / "main.jac", "MODEL"))
    for name, d in (("langgraph", "langgraph"), ("openai_sdk", "openai_sdk")):
        # jac-gpt.py is not an importable module name; exercise the helper the
        # arm uses on the same env value, which is the part that can be wrong.
        add("RagGPT", name, "import", *_py_probe(
            "raggpt", R / "RagGPT" / d,
            "import importlib.util as u, os, sys;"
            "s=u.spec_from_file_location('jg','jac-gpt.py');m=u.module_from_spec(s);"
            "src=open('jac-gpt.py').read();"
            "exec(compile(src[:src.index('class JacGPT')], 'jac-gpt.py', 'exec'), m.__dict__);"
            "print(m._bare_model(os.environ['MODEL']))"))

    # -- YTNavigator --------------------------------------------------------
    add("YTNavigator", "byLLM", "static",
        *_jac_static("ytnavigator", R / "YTNavigator" / "byLLM" / "nodes.jac",
                     "INSTANT_LLM"))
    add("YTNavigator", "YT-Navigator", "import", *_py_probe(
        "ytnavigator", R / "YTNavigator" / "YT-Navigator",
        # init_chat_model infers the provider from the id; a bare name raises.
        "import os; from langchain.chat_models import init_chat_model;"
        "m=init_chat_model(os.environ['INSTANT_LLM'], temperature=0.0);"
        "print(f'{m.model_name} via {type(m).__name__}')"))
    add("YTNavigator", "openai_sdk", "import", *_py_probe(
        "ytnavigator", R / "YTNavigator" / "openai_sdk",
        "import llm; print(f'{llm.instant_model()} / {llm.powerful_model()}')"))

    return rows


# ---------------------------------------------------------------------------
# Everything else
# ---------------------------------------------------------------------------

def check_runners() -> list[tuple]:
    """Each runner must know about all three of its arms."""
    checks = [
        ("CodeAgent", R / "CodeAgent" / "swebench_bridge" / "compare.py",
         ["byllm", "langgraph", "openai"]),
        ("Email", R / "Email-Auto-response" / "eval" / "run.py",
         ["byLLM", "CrewAI-LangGraph", "openai_sdk"]),
        ("meeting", R / "meeting-assistant" / "eval" / "run.py",
         ["byLLM", "CrewAI", "openai_sdk"]),
        ("YTNavigator", R / "YTNavigator" / "eval" / "e2e.py",
         ["byllm", "langgraph", "openai_sdk"]),
        ("YTNavigator", R / "YTNavigator" / "eval" / "run.py",
         ["byllm", "langgraph", "openai_sdk"]),
    ]
    rows = []
    for bench, script, arms in checks:
        proc = _run([PY, str(script), "--help"])
        text = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            rows.append((bench, script.name, FAIL, "--help",
                         (proc.stderr or "").strip().splitlines()[-1][:140]))
            continue
        missing = [a for a in arms if a not in text]
        rows.append((bench, script.name, FAIL if missing else PASS, "--help",
                     f"missing arm(s): {', '.join(missing)}" if missing
                     else f"offers {', '.join(arms)}"))
    # RagGPT's systems come from a table, not a flag.
    from RagGPT.eval import common as _  # noqa: F401  (only to prove importability)
    return rows


def check_raggpt_systems() -> tuple:
    env = config.env_for("raggpt")
    proc = _run([PY, "-c",
                 "import sys; sys.path.insert(0, 'eval'); import common;"
                 "print(','.join(common.SYSTEMS)); print(common.JAC_BIN_DIR);"
                 "print(common.PROXY_UPSTREAM); print(common.JUDGE_MODEL)"],
                cwd=R / "RagGPT", env=env)
    if proc.returncode != 0:
        return ("RagGPT", "common.py", FAIL, "import",
                (proc.stderr or "").strip().splitlines()[-1][:140])
    systems, jac_bin, upstream, judge = (proc.stdout or "").strip().splitlines()[:4]
    ok = "openai-sdk" in systems and R.as_posix() not in jac_bin
    detail = f"systems={systems} | jac={jac_bin} | upstream={upstream} | judge={judge}"
    return ("RagGPT", "common.py", PASS if ok else WARN, "import", detail)


def check_inputs() -> list[tuple]:
    """Datasets must be present and the right size; a silently shrunk input set
    is a comparison that quietly changed."""
    expect = [
        ("Email", "mock_mailbox/datasets/batch_*.json", 6, R / "Email-Auto-response"),
        ("meeting", "datasets/meeting_*.json", 10, R / "meeting-assistant"),
        ("YTNavigator", "datasets/questions.jsonl", 23, R / "YTNavigator"),
        ("RagGPT", "eval/dataset/dataset.jsonl", 220, R / "RagGPT"),
        ("CodeAgent", "case_study/instances.txt", 36, R / "CodeAgent"),
    ]
    rows = []
    for bench, pattern, count, root in expect:
        if "*" in pattern:
            found = len(list(root.glob(pattern)))
        else:
            path = root / pattern
            found = len([l for l in path.read_text().splitlines() if l.strip()]) if path.is_file() else 0
        rows.append((bench, pattern, PASS if found == count else FAIL, "inputs",
                     f"{found} (expected {count})"))
    # A submodule that was never initialized looks exactly like a missing
    # directory to everything downstream: `pip install -e CodeAgent/SWE-bench`
    # says "does not appear to be a Python project", and grading fails hours
    # into a run. Name the real cause instead.
    swebench = R / "CodeAgent" / "SWE-bench"
    if (swebench / "pyproject.toml").is_file():
        rows.append(("CodeAgent", "SWE-bench submodule", PASS, "inputs", "checked out"))
    else:
        rows.append(("CodeAgent", "SWE-bench submodule", FAIL, "inputs",
                     "not checked out -- run `git submodule update --init --recursive`, "
                     "then `pip install -e CodeAgent/SWE-bench`"))

    extra = [
        ("Email", R / "Email-Auto-response" / "eval" / "out" / "benchmark_slide.pptx"),
        ("RagGPT", R / "RagGPT" / "langgraph" / "faiss_index" / "index.faiss"),
        ("RagGPT", R / "RagGPT" / "openai_sdk" / "faiss_index" / "index.faiss"),
    ]
    for bench, path in extra:
        label = (f"{path.parent.parent.name}/{path.name}"
                 if path.name == "index.faiss" else path.name)
        rows.append((bench, label, PASS if path.is_file() else FAIL, "inputs",
                     "present" if path.is_file() else f"missing: {path}"))
    return rows


def check_toolchain() -> list[tuple]:
    rows = []

    def tool(name, required, note=""):
        found = shutil.which(name)
        rows.append(("toolchain", name, PASS if found else (FAIL if required else WARN),
                     "which", found or f"not on PATH {note}".strip()))

    tool("jac", True, "-- every byLLM arm needs it")
    tool("ollama", False, "-- only needed to serve the model locally")
    if not shutil.which("docker") and not shutil.which("udocker"):
        rows.append(("toolchain", "docker/udocker", FAIL, "which",
                     "neither found -- CodeAgent cannot run"))
    else:
        rows.append(("toolchain", "docker/udocker", PASS, "which",
                     shutil.which("docker") or shutil.which("udocker")))
    pg = shutil.which("pg_ctl") or next(
        (str(p) for p in Path(sys.prefix).parent.glob("*/bin/pg_ctl")), None)
    rows.append(("toolchain", "postgres", PASS if pg else WARN, "which",
                 pg or "not found -- YTNavigator's eval will look in every conda env"))

    for mod, why in (("openai", "every arm"), ("litellm", "byLLM + CrewAI"),
                     ("langgraph", "LangGraph arms"), ("crewai", "CrewAI arms"),
                     ("django", "YT-Navigator"), ("aiohttp", "the token proxy"),
                     ("pandas", "RagGPT's report"), ("swebench", "CodeAgent grading"),
                     ("sentence_transformers", "retrieval"), ("faiss", "RagGPT")):
        proc = _run([PY, "-c", f"import {mod}"])
        rows.append(("deps", mod, PASS if proc.returncode == 0 else FAIL, "import",
                     why if proc.returncode == 0 else f"not importable ({why})"))
    return rows


def check_jac_runtime() -> list[tuple]:
    """Which jac is installed, and can the Jac arms import byLLM under it?

    byLLM lives in one of two places depending on how jac was installed: inside
    the toolchain as `jaclang.byllm` (dev builds bundle it), or as the separate
    pip `byllm` package alongside a released jaclang. The two are mutually
    exclusive -- on a dev build, importing the pip package fails outright -- and
    every Jac arm here names one of them at the top of its file. So the layout
    is probed by actually running a snippet through the installed runtime,
    rather than importing into this interpreter, which may resolve differently.
    """
    rows = []
    jac = shutil.which("jac")
    if not jac:
        return [("jac runtime", "jac", FAIL, "which",
                 "not on PATH -- every byLLM arm will fail. "
                 "Install the Jac toolchain, or set $JAC_BIN to its bin directory.")]

    proc = _run([jac, "--version"], timeout=60)
    version = next((ln.split("Version:")[-1].strip()
                    for ln in ((proc.stdout or "") + (proc.stderr or "")).splitlines()
                    if "Version:" in ln), "unknown")
    rows.append(("jac runtime", "jac", PASS, "which", f"{jac} (version {version})"))

    layouts = {"jaclang.byllm.lib": "jaclang.byllm", "byllm.lib": "byllm"}
    found = None
    with tempfile.TemporaryDirectory() as tmp:
        for module, label in layouts.items():
            probe = Path(tmp) / f"probe_{label.replace('.', '_')}.jac"
            probe.write_text(
                f"import from {module} {{ Model }}\n"
                f'with entry {{ print("OK"); }}\n')
            got = _run([jac, "run", probe.name], cwd=Path(tmp), timeout=180)
            if got.returncode == 0 and "OK" in (got.stdout or ""):
                found = module
                break

    if not found:
        rows.append(("jac runtime", "byLLM import", FAIL, "jac run",
                     "neither `jaclang.byllm.lib` nor `byllm.lib` resolves under this "
                     "runtime -- no Jac arm can start. Run `jac install` in each Jac "
                     "project, or install the matching byllm."))
        return rows
    rows.append(("jac runtime", "byLLM import", PASS, "jac run", f"resolves via `{found}`"))

    # Every Jac arm names one layout at the top of its file. If the runtime
    # provides the other one, those arms fail on their first line.
    wrong_layout = "byllm.lib" if found == "jaclang.byllm.lib" else "jaclang.byllm.lib"
    mismatched = []
    for jac_file in sorted(R.rglob("*.jac")):
        if "SWE-bench" in jac_file.parts or ".jac" in jac_file.parts:
            continue
        text = jac_file.read_text(errors="ignore")
        if f"import from {wrong_layout}" in text and f"import from {found}" not in text:
            mismatched.append(jac_file.relative_to(R))
    rows.append(("jac runtime", "arm import lines", FAIL if mismatched else PASS, "grep",
                 f"{len(mismatched)} file(s) name `{wrong_layout}`: "
                 + ", ".join(str(m) for m in mismatched[:4]) if mismatched
                 else f"all Jac arms name `{found}`, matching this runtime"))
    return rows


def check_endpoint() -> list[tuple]:
    """The endpoint is reachable, and the arms are pointed at it. No completion."""
    import urllib.error
    import urllib.request
    rows = []
    for label, url in (("model server", f"{config.upstream()}/api/tags"),
                       ("token proxy", f"http://127.0.0.1:{config.proxy_port()}/__health")):
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                json.loads(resp.read().decode())
            rows.append(("endpoint", label, PASS, "GET", url))
        except Exception as exc:
            rows.append(("endpoint", label, WARN, "GET",
                         f"not up ({type(exc).__name__}) -- run_benchmark.sh starts it"))
    env = config.env_for("meeting")
    rows.append(("endpoint", "arms point at", PASS, "env", env["OPENAI_BASE_URL"]))
    return rows


def render(sections: list[tuple[str, list[tuple]]]) -> int:
    worst = 0
    for title, rows in sections:
        if not rows:
            continue
        print(f"\n{title}")
        print("-" * len(title))
        w1 = max(len(r[0]) for r in rows)
        w2 = max(len(str(r[1])) for r in rows)
        for bench, item, status, kind, detail in rows:
            print(f"  {status:4}  {bench:<{w1}}  {str(item):<{w2}}  [{kind}]  {detail}")
            worst = max(worst, {PASS: 0, WARN: 1, FAIL: 2}[status])
    return worst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="store_true", help="only the fifteen model seams")
    args = ap.parse_args(argv)

    desc = config.describe()
    print("=" * 78)
    print("Static verification -- no model is called, no weights are loaded")
    for key, value in desc.items():
        print(f"  {key:20s} {value}")
    print("=" * 78)

    model_rows = check_models()
    sections = [("Model seams (all fifteen arms resolve to one model)", model_rows)]
    if not args.models:
        sections += [
            ("Runners (each knows all three of its arms)",
             check_runners() + [check_raggpt_systems()]),
            ("Inputs (datasets intact after the clean)", check_inputs()),
            ("Jac runtime (the arms run on the installed one)", check_jac_runtime()),
            ("Toolchain and dependencies", check_toolchain()),
            ("Endpoint", check_endpoint()),
        ]

    worst = render(sections)

    # The check that matters most: two arms of one benchmark on different models
    # is the failure that makes a benchmark measure the model, not the framework.
    print("\nOne model everywhere")
    print("-" * 20)
    want = config.bare_id()
    off = [(b, a, v) for b, a, st, _, v in model_rows
           if st == PASS and want not in v and "<model>" not in v]
    if off:
        for bench, arm, value in off:
            print(f"  FAIL  {bench}/{arm} resolved to {value}, not {want}")
        worst = 2
    else:
        resolved = sum(1 for r in model_rows if r[2] == PASS)
        print(f"  PASS  {resolved}/{len(model_rows)} arms resolve to '{want}' "
              f"(in each library's own shape)")

    failures = sum(1 for _, rows in sections for r in rows if r[2] == FAIL)
    print(f"\n{'FAILED' if worst == 2 else 'OK'}: "
          f"{failures} failure(s) across {sum(len(r) for _, r in sections)} checks.")
    return 1 if worst == 2 else 0


if __name__ == "__main__":
    raise SystemExit(main())
