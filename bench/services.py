"""Bring up the one model server and the one token ledger the whole sweep uses.

The benchmark is "batched" in the sense that matters here: fifteen arms across
five projects share a single long-lived endpoint instead of each managing its
own. Arms still run one at a time -- they are being compared on tokens and
wall-clock, and two arms in flight would contend for the same GPU and make both
numbers meaningless.

Between the arms and the model sits `RagGPT/eval/harness/proxy.py`, already
written as a general OpenAI-compatible logging reverse proxy. Pointing every
arm at it gives one file recording every call from byLLM, LangGraph, CrewAI and
the raw SDK alike, with identical accounting, without touching any arm's code.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config

PROXY_SCRIPT = config.REPO_ROOT / "RagGPT" / "eval" / "harness" / "proxy.py"


def _log(*args) -> None:
    """Progress goes to stderr.

    run_benchmark.sh reads the served model tag off this module's stdout, so a
    log line on stdout would be captured as part of the model name.
    """
    print(*args, file=sys.stderr)


def _get_json(url: str, timeout: float = 3.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _post_json(url: str, payload: dict, timeout: float = 600.0):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def ollama_root() -> str:
    return config.upstream()


def ollama_up() -> bool:
    try:
        _get_json(f"{ollama_root()}/api/tags")
        return True
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return False


def ensure_ollama(timeout: float = 60.0) -> subprocess.Popen | None:
    """Attach to a running server, else start one. Returns the process we own."""
    if ollama_up():
        _log(f"ollama: already running at {ollama_root()}")
        return None
    if not shutil.which("ollama"):
        raise SystemExit(
            f"no ollama server at {ollama_root()} and no `ollama` on PATH.\n"
            "Install it (https://ollama.com/download), or point BENCH_BASE_URL "
            "at another OpenAI-compatible server (vLLM, llama.cpp, ...)."
        )
    proc = subprocess.Popen(["ollama", "serve"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ollama_up():
            _log(f"ollama: started at {ollama_root()}")
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise SystemExit(f"ollama did not come up at {ollama_root()} within {timeout:.0f}s")


def _tags() -> set[str]:
    try:
        return {m["name"] for m in _get_json(f"{ollama_root()}/api/tags").get("models", [])}
    except Exception:
        return set()


def _has_model(name: str) -> bool:
    tags = _tags()
    return name in tags or f"{name}:latest" in tags


def _create_derived(base: str, derived: str, ctx: int) -> str:
    """Make `derived` = `base` with a bigger num_ctx. Returns "" on success.

    Three ways, because they are not equally available. The HTTP API goes first:
    `ollama create -f <path>` makes the *daemon* read that path, so a Modelfile
    in a private temp directory can fail with "no Modelfile or safetensors files
    found" even though the file plainly exists -- the daemon runs as its own
    user and cannot see it. The API carries the content in the request instead
    and sidesteps the question.
    """
    attempts: list[str] = []

    for payload in (
        # Current shape (ollama >= ~0.20).
        {"model": derived, "from": base, "parameters": {"num_ctx": ctx}, "stream": False},
        # Older shape, for builds that do not understand `from`/`parameters`.
        {"name": derived, "modelfile": f"FROM {base}\nPARAMETER num_ctx {ctx}\n",
         "stream": False},
    ):
        try:
            result = _post_json(f"{ollama_root()}/api/create", payload, timeout=900)
            if isinstance(result, dict) and result.get("error"):
                attempts.append(f"api: {result['error'][:80]}")
                continue
            if _has_model(derived):
                return ""
            attempts.append("api: reported success but the model is not listed")
        except Exception as exc:
            attempts.append(f"api: {type(exc).__name__}")

    # Last resort: the CLI, with a world-readable Modelfile so a daemon running
    # as another user can still read it.
    if shutil.which("ollama"):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                Path(tmp).chmod(0o755)
                modelfile = Path(tmp) / "Modelfile"
                modelfile.write_text(f"FROM {base}\nPARAMETER num_ctx {ctx}\n")
                modelfile.chmod(0o644)
                proc = subprocess.run(["ollama", "create", derived, "-f", str(modelfile)],
                                      capture_output=True, text=True, timeout=900)
            if proc.returncode == 0:
                return ""
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            attempts.append(f"cli: {tail[-1][:80] if tail else 'failed'}")
        except Exception as exc:
            attempts.append(f"cli: {type(exc).__name__}")

    return "; ".join(attempts)

def ensure_model() -> str:
    """Pull the model, then derive a long-context variant. Returns the served id.

    Ollama's `num_ctx` defaults to 4096 and cannot be set over the /v1
    endpoint, but CodeAgent alone asks for a 16k completion. So we bake the
    window into a derived model with `ollama create`. Everything downstream
    talks to that tag.
    """
    base = config.model()
    if not _has_model(base):
        if not shutil.which("ollama"):
            raise SystemExit(f"model {base!r} is not served and `ollama` is not on PATH to pull it")
        _log(f"ollama: pulling {base} (this is a large download the first time)")
        if subprocess.run(["ollama", "pull", base]).returncode != 0:
            raise SystemExit(f"`ollama pull {base}` failed")

    ctx = config.ctx_window()
    derived = f"{base.split(':')[0]}-bench{ctx // 1024}k"
    if not shutil.which("ollama"):
        _log(f"ollama: no CLI to derive a {ctx}-token variant; serving {base} as-is")
        return base
    if not _has_model(derived):
        _log(f"ollama: creating {derived} (num_ctx {ctx})")
        errors = _create_derived(base, derived, ctx)
        if errors:
            _log(f"ollama: could not derive {derived} ({errors}); serving {base} with "
                 f"its default context window (usually 4096 -- far too small for "
                 f"CodeAgent's tool transcripts). Lower BENCH_CTX and re-run, or set "
                 f"OLLAMA_CONTEXT_LENGTH on the ollama service.")
            return base
    _log(f"ollama: serving {derived}")
    return derived


# ---------------------------------------------------------------------------
# The token ledger
# ---------------------------------------------------------------------------

def proxy_health_url() -> str:
    return f"http://127.0.0.1:{config.proxy_port()}/__health"


def proxy_health() -> dict | None:
    try:
        return _get_json(proxy_health_url(), timeout=2.0)
    except Exception:
        return None


def ensure_proxy(log_path: Path, timeout: float = 30.0) -> subprocess.Popen | None:
    """Start proxy.py in front of the model server, logging to `log_path`.

    RagGPT's own runner also calls a proxy into being, and it verifies the one
    it finds is forwarding where it expects before trusting it -- so a proxy
    already pointed at our upstream with our log is reused by both.
    """
    want_upstream, want_log = config.upstream(), str(log_path)
    health = proxy_health()
    if health:
        if health.get("upstream") == want_upstream and health.get("log") == want_log:
            _log(f"proxy: already running -> {want_upstream}")
            return None
        _log(f"proxy: replacing a proxy pointed at {health.get('upstream')}")
        _kill_proxies()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "PROXY_UPSTREAM": want_upstream,
        "PROXY_PORT": str(config.proxy_port()),
        "PROXY_LOG": want_log,
    })
    proc = subprocess.Popen([sys.executable, str(PROXY_SCRIPT)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    deadline = time.time() + timeout
    while time.time() < deadline:
        health = proxy_health()
        if health and health.get("upstream") == want_upstream:
            _log(f"proxy: 127.0.0.1:{config.proxy_port()} -> {want_upstream}  log: {want_log}")
            return proc
        time.sleep(0.4)
    proc.terminate()
    raise SystemExit(f"proxy did not come up on port {config.proxy_port()}")


def _kill_proxies() -> None:
    subprocess.run(["pkill", "-f", str(PROXY_SCRIPT)], capture_output=True)
    time.sleep(1)


def write_pricing_table(path: Path) -> None:
    """A zero-cost entry for the served model.

    Email's scorer reprices runs from a table keyed by OpenAI model names. With
    no entry, every local run is reported under `cost.unpriced_models`; with a
    zero entry it reports what is true -- these tokens cost nothing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    names = {config.served_model(), config.bare_id(), config.litellm_id(), config.model()}
    path.write_text(json.dumps({n: [0.0, 0.0, 0.0] for n in sorted(names)}, indent=2))


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------

_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]

_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "Person",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
            "additionalProperties": False,
        },
    },
}


def placement(model: str) -> dict:
    """Where a loaded model is actually running: GPU, CPU, or split across both.

    Ollama falls back to CPU silently when it cannot fit the weights in VRAM --
    no error, no warning, just a model that answers perhaps thirty times slower.
    On a benchmark whose largest stage is already hours on a GPU, that is the
    difference between a run and a week.

    Read from /api/ps (size_vram against size), falling back to the PROCESSOR
    column of `ollama ps`. Returns fraction=None rather than guessing when
    neither is readable, so this can only ever report, never false-alarm.
    """
    out = {"fraction": None, "detail": "", "source": ""}
    try:
        loaded = (_get_json(f"{ollama_root()}/api/ps", timeout=5) or {}).get("models") or []
    except Exception:
        loaded = []
    for entry in loaded:
        name = entry.get("name") or entry.get("model") or ""
        if name.split(":")[0] != model.split(":")[0]:
            continue
        total, vram = entry.get("size"), entry.get("size_vram")
        if isinstance(total, int) and isinstance(vram, int) and total > 0:
            out.update(fraction=vram / total, source="/api/ps",
                       detail=f"{vram / 1e9:.1f} GB of {total / 1e9:.1f} GB in VRAM")
            return out

    if shutil.which("ollama"):
        proc = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=30)
        for line in (proc.stdout or "").splitlines():
            if not line.startswith(model.split(":")[0]):
                continue
            for token in line.split():
                if token.endswith("%"):
                    try:
                        pct = float(token.rstrip("%"))
                    except ValueError:
                        continue
                    is_gpu = "GPU" in line.upper().split(token)[-1][:8] or "GPU" in line.upper()
                    out.update(fraction=(pct / 100) if is_gpu else 0.0,
                               source="ollama ps", detail=line.strip())
                    return out
    return out

def probe(endpoint: str | None = None, model: str | None = None) -> dict:
    """Check the served model can do what all five benchmarks require.

    Every project in this repo drives the model with tool calls, and four of
    them also ask for a strict JSON schema. A model that cannot do those will
    produce a table of zeros after several hours; thirty seconds here says so
    up front.
    """
    endpoint = (endpoint or config.base_url()).rstrip("/")
    model = model or config.served_model()
    url = f"{endpoint}/chat/completions"
    out: dict = {"endpoint": endpoint, "model": model}

    try:
        r = _post_json(url, {"model": model,
                             "messages": [{"role": "user", "content": "Reply with the word: ok"}],
                             "max_tokens": 16}, timeout=300)
    except Exception as exc:
        out["reachable"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["reachable"] = True
    # The completion above forced the load, so placement is readable now.
    out["placement"] = placement(model)
    usage = r.get("usage") or {}
    out["reports_usage"] = bool(usage.get("total_tokens") or usage.get("prompt_tokens"))
    out["usage"] = usage

    try:
        r = _post_json(url, {"model": model,
                             "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
                             "tools": _TOOL, "tool_choice": "auto"}, timeout=300)
        msg = r["choices"][0]["message"]
        out["tool_calls"] = bool(msg.get("tool_calls"))
        if not out["tool_calls"]:
            content = (msg.get("content") or "").strip()
            # The usual local-model failure: the model names the function in
            # prose instead of emitting a tool_calls block. Every arm here
            # dispatches on the structured field, so this counts as no tools.
            if "get_weather" in content:
                out["tool_calls_error"] = (
                    "the model described the call in its content instead of "
                    f"emitting tool_calls: {content[:120]!r}")
            else:
                out["tool_calls_error"] = f"no tool_calls in the reply: {content[:120]!r}"
    except Exception as exc:
        out["tool_calls"] = False
        out["tool_calls_error"] = f"{type(exc).__name__}: {exc}"

    try:
        r = _post_json(url, {"model": model,
                             "messages": [{"role": "user",
                                           "content": "Ada Lovelace was 36. Return name and age."}],
                             "response_format": _SCHEMA}, timeout=300)
        content = r["choices"][0]["message"].get("content") or ""
        parsed = json.loads(content)
        out["json_schema"] = isinstance(parsed, dict) and "name" in parsed and "age" in parsed
    except Exception as exc:
        out["json_schema"] = False
        out["json_schema_error"] = f"{type(exc).__name__}: {exc}"

    try:
        info = _post_json(f"{ollama_root()}/api/show", {"model": model}, timeout=30)
        params = info.get("parameters") or ""
        out["num_ctx"] = next(
            (line.split()[-1] for line in params.splitlines() if line.startswith("num_ctx")),
            "(model default)")
    except Exception:
        out["num_ctx"] = "(unknown)"

    out["ok"] = bool(out.get("reachable") and out.get("reports_usage") and out.get("tool_calls"))
    return out


def print_probe(result: dict) -> bool:
    def mark(v):
        return "PASS" if v else "FAIL"
    print(f"\nProbe: {result['model']} at {result['endpoint']}")
    if not result.get("reachable"):
        print(f"  reachable            FAIL  {result.get('error', '')}")
        return False
    print(f"  reachable            PASS")
    print(f"  reports usage        {mark(result.get('reports_usage'))}  {result.get('usage')}")
    print(f"  tool calling         {mark(result.get('tool_calls'))}  "
          f"{result.get('tool_calls_error', '')}")
    print(f"  strict json_schema   {mark(result.get('json_schema'))}  "
          f"{result.get('json_schema_error', '')}")
    print(f"  num_ctx              {result.get('num_ctx')}")
    place = result.get("placement") or {}
    frac, detail = place.get("fraction"), place.get("detail")
    if frac is None:
        print(f"  running on           (could not read placement)")
    elif frac >= 0.99:
        print(f"  running on           GPU  {detail}")
    elif frac <= 0.01:
        print(f"  running on           CPU  {detail}   <-- SEE BELOW")
    else:
        print(f"  running on           {frac * 100:.0f}% GPU / {(1 - frac) * 100:.0f}% CPU  "
              f"{detail}   <-- SEE BELOW")
    if not result.get("tool_calls"):
        print("\n  Tool calling is required by all five benchmarks. Expect near-zero\n"
              "  scores across the board; this measures the model, not the frameworks.")
    if not result.get("json_schema"):
        print("\n  Strict JSON schema is unsupported. The arms have text fallbacks, so the\n"
              "  run will complete -- watch the parse-failure rate in the report.")
    if frac is not None and frac < 0.99:
        where = "entirely on CPU" if frac <= 0.01 else f"only {frac * 100:.0f}% on the GPU"
        print(f"\n  The model is running {where}. Ollama falls back to CPU silently when\n"
              "  the weights do not fit in VRAM -- no error, just a model perhaps thirty\n"
              "  times slower. CodeAgent alone is hours on a GPU; on CPU this sweep does\n"
              "  not finish in any useful time. Usual causes:\n"
              "    * the GPU is not visible to the ollama service (`nvidia-smi` works for\n"
              "      you but not for the service -- check `systemctl status ollama`)\n"
              "    * the model plus $BENCH_CTX exceeds VRAM; lower BENCH_CTX or use a\n"
              "      smaller model\n"
              "    * something else already holds the VRAM (`nvidia-smi`)")
    return bool(result.get("ok"))


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", action="store_true", help="check model capabilities and exit")
    ap.add_argument("--start", action="store_true", help="start ollama + the proxy and exit")
    ap.add_argument("--log", default=str(config.RUNS_ROOT / "probe" / "tokens.jsonl"))
    args = ap.parse_args(argv)

    ensure_ollama()
    # run_benchmark.sh derives the long-context model once and exports the tag;
    # deriving it again here would repeat a slow step and scramble the output.
    served = os.environ.get("BENCH_SERVED_MODEL") or ensure_model()
    os.environ["BENCH_SERVED_MODEL"] = served

    if args.start:
        ensure_proxy(Path(args.log))
    if args.probe or not args.start:
        return 0 if print_probe(probe(model=served)) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
