"""The model seam: one local endpoint, fifteen arms.

Every implementation in this repo reaches a model through one of three
libraries, and each wants the model id in a different shape:

    litellm      (byLLM `Model`, CrewAI `LLM`)   openai/<name>
    langchain    (`init_chat_model`)             openai:<name>
    openai SDK   (`OpenAI()`, `ChatOpenAI`)      <name>

Two projects already normalize between those shapes on their own -- Email's
three arms agree on `OPENAI_MODEL_NAME` (byLLM prefixes, the SDK strips) and
YTNavigator's on `INSTANT_LLM`/`POWERFUL_LLM` (byLLM rewrites ':' to '/', the
SDK drops the provider). For those two, one env value serves all three arms
unchanged. The other three projects had no normalization, so `bench/` adds the
same two helpers there rather than inventing a per-arm variable: a benchmark
whose arms read different knobs is a benchmark whose arms can silently diverge
on model, which measures the model instead of the framework.

Nothing here talks to a network. `env_for()` returns a dict; the caller decides
which subprocess gets it.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = Path(__file__).resolve().parent / "runs"

# ---------------------------------------------------------------------------
# Knobs. Every one has a default, so `run_benchmark.sh` with no arguments runs.
# ---------------------------------------------------------------------------

#: The model to pull and serve.
DEFAULT_MODEL = "muse-glimmer"
#: Ollama's OpenAI-compatible endpoint.
DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
#: Where the token-counting proxy listens. Every arm is pointed here, not at
#: the model server, so one log records every call from every framework.
DEFAULT_PROXY_PORT = 8899
#: Ollama defaults num_ctx to 4096, which is far too small for CodeAgent's tool
#: transcripts. services.py bakes this into a derived model.
DEFAULT_CTX = 32768


def model() -> str:
    """The model to pull and serve."""
    return os.environ.get("BENCH_MODEL", DEFAULT_MODEL)


def served_model() -> str:
    """The model id to put on the wire.

    Differs from `model()` when services.py derived a long-context variant --
    `ollama create <name>-bench` from a Modelfile with a bigger `num_ctx`,
    because that parameter cannot be set over the /v1 endpoint.
    """
    return os.environ.get("BENCH_SERVED_MODEL") or model()


def judge_model() -> str:
    """The judge runs on the same local model: the run is fully offline."""
    return os.environ.get("BENCH_JUDGE_MODEL") or served_model()


def ctx_window() -> int:
    return int(os.environ.get("BENCH_CTX", DEFAULT_CTX))


def base_url() -> str:
    """Where the model actually is (what the proxy forwards to)."""
    return os.environ.get("BENCH_BASE_URL", DEFAULT_BASE_URL)


def upstream() -> str:
    """`base_url()` without the /v1 suffix -- what proxy.py wants."""
    return base_url().rstrip("/").removesuffix("/v1").rstrip("/")


def proxy_port() -> int:
    return int(os.environ.get("BENCH_PROXY_PORT", DEFAULT_PROXY_PORT))


def proxy_base_url() -> str:
    return f"http://127.0.0.1:{proxy_port()}/v1"


def run_id() -> str:
    """This sweep's id. run_benchmark.sh exports it before anything starts."""
    return os.environ.get("BENCH_RUN_ID", "adhoc")


def ledger_path() -> Path:
    """The one token log. RagGPT's scorer joins tokens from whatever
    $PROXY_LOG names, so pointing it here is what lets that benchmark share the
    ledger instead of keeping a second one."""
    return RUNS_ROOT / run_id() / "tokens.jsonl"


def use_proxy() -> bool:
    """Set BENCH_NO_PROXY=1 to point the arms straight at the model server."""
    return os.environ.get("BENCH_NO_PROXY", "") not in ("1", "true", "yes")


def endpoint_for_arms() -> str:
    return proxy_base_url() if use_proxy() else base_url()


# ---------------------------------------------------------------------------
# Model-id shapes
# ---------------------------------------------------------------------------

def litellm_id(name: str | None = None) -> str:
    """`openai/<name>` -- byLLM and CrewAI both resolve through litellm, which
    cannot infer a provider from an unknown bare name."""
    name = name or served_model()
    return name if "/" in name else f"openai/{name}"


def langchain_id(name: str | None = None) -> str:
    """`openai:<name>` -- what `init_chat_model` needs to pick ChatOpenAI (and
    therefore to honour OPENAI_BASE_URL) for a name it has never seen."""
    name = name or served_model()
    if ":" in name:
        return name
    return f"openai:{name.split('/')[-1]}"


def bare_id(name: str | None = None) -> str:
    """`<name>` -- what the raw SDK and a local server want on the wire."""
    name = name or served_model()
    return name.replace(":", "/").split("/")[-1]


# ---------------------------------------------------------------------------
# Toolchain discovery -- never a hardcoded absolute path, or the remote box
# breaks before the first token.
# ---------------------------------------------------------------------------

def jac_bin_dir() -> str:
    """Directory holding the `jac` binary, for PATH-prepending."""
    explicit = os.environ.get("JAC_BIN", "")
    if explicit and Path(explicit).is_dir():
        return explicit
    found = shutil.which("jac")
    if found:
        return str(Path(found).resolve().parent)
    return ""


# ---------------------------------------------------------------------------
# The environment each arm runs under
# ---------------------------------------------------------------------------

def base_env() -> dict[str, str]:
    """What every arm of every project gets, whatever its framework."""
    env = dict(os.environ)
    endpoint = endpoint_for_arms()
    env.update({
        # The raw SDK, langchain-openai and litellm each read one of these.
        "OPENAI_BASE_URL": endpoint,
        "OPENAI_API_BASE": endpoint,
        # A dummy key is not optional: four preflights in this repo hard-exit on
        # an empty OPENAI_API_KEY before any request is made.
        "OPENAI_API_KEY": os.environ.get("BENCH_API_KEY", "ollama"),
        # Judges: Email's and meeting's scorers read this; the others take a flag.
        "EVAL_JUDGE_MODEL": judge_model(),
        # Email prices its runs from a table keyed by OpenAI model names. Without
        # an override every local run reports as "unpriced"; a zero-cost entry
        # says what is true -- the tokens were free.
        "EVAL_PRICING_FILE": str(RUNS_ROOT / "pricing.json"),
        # byLLM's auto-compaction asks litellm for the context window and gets
        # nothing for a non-OpenAI id. Told explicitly, it compacts correctly.
        "BENCH_CTX": str(ctx_window()),
        "BYLLM_CTX_WINDOW": str(ctx_window()),
        # The proxy's own configuration travels with the arms, because RagGPT's
        # runner reads it too: it checks a running proxy forwards where this run
        # expects before reusing it, and its scorer joins tokens from this log.
        "PROXY_UPSTREAM": upstream(),
        "PROXY_PORT": str(proxy_port()),
        "BENCH_PROXY_PORT": str(proxy_port()),
        "PROXY_LOG": str(ledger_path()),
    })
    jac_dir = jac_bin_dir()
    if jac_dir:
        env["JAC_BIN"] = jac_dir
        if jac_dir not in env.get("PATH", "").split(os.pathsep):
            env["PATH"] = jac_dir + os.pathsep + env.get("PATH", "")
    # Streaming carries usage only via stream_options.include_usage, which not
    # every local server accepts, and CodeAgent's retry shim does not catch a
    # rejection. Off by default locally; BENCH_STREAM=1 restores it.
    env["CODEAGENT_STREAM"] = os.environ.get("BENCH_STREAM", "0")
    return env


#: Which env var each project's arms read for the model. See the module
#: docstring for why one variable can serve three arms.
PROJECTS = ("codeagent", "email", "meeting", "raggpt", "ytnavigator")


def env_for(project: str) -> dict[str, str]:
    """The full environment for one project's runner."""
    if project not in PROJECTS:
        raise ValueError(f"unknown project {project!r}; known: {', '.join(PROJECTS)}")
    env = base_env()

    if project == "codeagent":
        # All three arms read CODEAGENT_MODEL; each normalizes the shape itself.
        env["CODEAGENT_MODEL"] = served_model()
        env.setdefault("CODEAGENT_MAX_TOKENS", str(min(ctx_window() // 2, 16384)))

    elif project == "email":
        # byLLM prefixes 'openai/' when absent, the SDK strips any prefix, and
        # CrewAI hands the value straight to litellm -- so the litellm shape is
        # the one value all three read correctly.
        env["OPENAI_MODEL_NAME"] = litellm_id()

    elif project == "meeting":
        env["MEETING_MODEL"] = served_model()

    elif project == "raggpt":
        env["MODEL"] = served_model()
        env["BENCH_JUDGE_MODEL"] = judge_model()

    elif project == "ytnavigator":
        # byLLM rewrites ':' to '/', the SDK drops the provider, and the Django
        # app needs the colon form for init_chat_model's provider inference.
        env["INSTANT_LLM"] = langchain_id()
        env["POWERFUL_LLM"] = langchain_id()
        # This project's judge goes through litellm rather than the raw SDK, so
        # it needs the prefixed form the others must not get.
        env["EVAL_JUDGE_MODEL"] = litellm_id(judge_model())

    return env


def describe() -> dict[str, str]:
    """What this run is, for the report header."""
    return {
        "model": model(),
        "served_model": served_model(),
        "judge_model": judge_model(),
        "base_url": base_url(),
        "endpoint_for_arms": endpoint_for_arms(),
        "ctx_window": str(ctx_window()),
        "jac_bin_dir": jac_bin_dir() or "(not found)",
    }
