"""Shared paths, environment setup, and helpers for the Jac-GPT eval.

Run everything with the `jaseci` conda python:
    /home/xiaoyu/miniconda3/envs/jaseci/bin/python
"""

import json
import os
import re
import subprocess
import uuid
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
CODER_DIR = EVAL_DIR.parent

JAC_BIN_DIR = "/home/xiaoyu/jaseci/jac/zig-out/bin"

DATASET_PATH = EVAL_DIR / "dataset" / "dataset.jsonl"
DATASET_REPORT_PATH = EVAL_DIR / "dataset" / "dataset_report.md"
RESULTS_DIR = EVAL_DIR / "results"
RAW_RUNS_PATH = RESULTS_DIR / "raw_runs.jsonl"
PROXY_LOG_PATH = RESULTS_DIR / "proxy_log.jsonl"
JUDGED_PATH = RESULTS_DIR / "judged.jsonl"
JAC_CHECK_ENV = EVAL_DIR / "harness" / "jac_check_env"

PROXY_PORT = 8899
PROXY_BASE_URL = f"http://127.0.0.1:{PROXY_PORT}/v1"

# The three systems under test. `kind` picks the driver in harness/drivers.py.
SYSTEMS = {
    "langgraph": {"kind": "python", "dir": CODER_DIR / "langgraph"},
    "jac": {"kind": "jac", "dir": CODER_DIR / "Jac-Rag-GPT", "port": 8501},
    "jac-byllm-router": {"kind": "jac", "dir": CODER_DIR / "Jac-Rag-GPT-ByllmRouter", "port": 8502},
}

AGENTS = ["RagChat", "CodingChat", "DebuggerChat", "QAChat", "OffTopicChat"]

CATEGORIES = ["rag_qa", "coding", "debugging", "small_talk", "off_topic", "multi_turn"]

# Synthesis + judging model (the systems under test stay on their configured gpt-4.1-mini).
JUDGE_MODEL = "gpt-4.1"


def setup_env() -> None:
    """Prepend the dev jac binary to PATH and load Coder/.env into os.environ."""
    path = os.environ.get("PATH", "")
    if JAC_BIN_DIR not in path.split(os.pathsep):
        os.environ["PATH"] = JAC_BIN_DIR + os.pathsep + path
    env_file = CODER_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_api_key() -> str:
    setup_env()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit(
            f"OPENAI_API_KEY is not set. Put it in {CODER_DIR / '.env'} "
            "(OPENAI_API_KEY=sk-...) and rerun."
        )
    return key


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


JAC_FENCE_RE = re.compile(r"```(?:jac|jaclang)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
ANY_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n(.*?)```", re.DOTALL)


def extract_jac_code(text: str) -> str | None:
    """Return the last fenced Jac code block in `text` (falling back to any fence)."""
    blocks = JAC_FENCE_RE.findall(text) or ANY_FENCE_RE.findall(text)
    return blocks[-1].strip() if blocks else None


def jac_check(code: str, timeout: int = 60) -> tuple[bool, str]:
    """Compile-check a Jac snippet inside jac_check_env. Returns (passed, output)."""
    setup_env()
    name = f"snippet_{uuid.uuid4().hex[:12]}.jac"
    path = JAC_CHECK_ENV / name
    path.write_text(code if code.endswith("\n") else code + "\n")
    try:
        proc = subprocess.run(
            ["jac", "check", name],
            cwd=JAC_CHECK_ENV,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode == 0, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return False, f"jac check timed out after {timeout}s"
    finally:
        path.unlink(missing_ok=True)
