#!/usr/bin/env bash
#
# The whole comparison, one command: five benchmarks x three arms (Jac/byLLM,
# an agent framework, a plain OpenAI-SDK baseline) against one locally served
# model.
#
#   ./run_benchmark.sh --smoke              # ~minutes: proves the wiring
#   ./run_benchmark.sh                      # the full sweep
#   ./run_benchmark.sh --clean              # delete previous results first
#   ./run_benchmark.sh --verify-only        # check the wiring; calls no model
#   ./run_benchmark.sh --probe-only         # check the model can do this
#   ./run_benchmark.sh --skip codeagent     # everything but SWE-bench
#
# Knobs (all optional, all with defaults):
#
#   BENCH_MODEL        model to pull and serve         (default: muse-glimmer)
#   BENCH_BASE_URL     an OpenAI-compatible endpoint   (default: ollama on :11434)
#   BENCH_CTX          context window to bake in       (default: 32768)
#   BENCH_JUDGE_MODEL  judge model                     (default: the same one)
#   BENCH_CONDA_ENV    environment to run in           (default: jaseci)
#   BENCH_NO_PROXY=1   skip the token ledger and talk to the server directly
#
# Point BENCH_BASE_URL at vLLM, llama.cpp or anything else that speaks
# /v1/chat/completions and everything below still holds -- only the model
# pull/derive step is ollama-specific, and it is skipped when the model is
# already being served.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-jaseci}"
DO_CLEAN=0
PROBE_ONLY=0
VERIFY_ONLY=0
SWEEP_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)       DO_CLEAN=1; shift ;;
        --probe-only)  PROBE_ONLY=1; shift ;;
        --verify-only) VERIFY_ONLY=1; shift ;;
        -h|--help)    grep -m1 -B999 '^set -euo' "${BASH_SOURCE[0]}" \
                          | grep '^#' | tail -n +2 | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)            SWEEP_ARGS+=("$1"); shift ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$candidate/etc/profile.d/conda.sh"
        break
    fi
done

if command -v conda >/dev/null 2>&1 && conda env list | grep -qE "^${BENCH_CONDA_ENV}\s"; then
    conda activate "$BENCH_CONDA_ENV"
    echo "env: conda '$BENCH_CONDA_ENV' -> $(python -V 2>&1)"
else
    echo "env: conda environment '$BENCH_CONDA_ENV' not found; using $(command -v python3)" >&2
    echo "     (create it from requirements.txt -- see BENCHMARK.md)" >&2
fi
PY="$(command -v python || command -v python3)"

# A wrong interpreter fails later as a pile of missing packages rather than as
# one version problem, so say it here instead.
"$PY" - <<'PYCHECK' || exit 1
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 12):
    sys.exit(f"Python {major}.{minor} is too old: byllm, scipy and others publish "
             f"no build for it.\nCreate the environment first -- "
             f"conda create -n jaseci python=3.13 && conda activate jaseci")
if (major, minor) >= (3, 14):
    sys.exit(f"Python {major}.{minor} is too new: both CrewAI arms require <3.14.")
PYCHECK

# The Jac arms need `jac`, and two evals used to hardcode an absolute path to
# it. They now discover it, but it still has to be findable.
if ! command -v jac >/dev/null 2>&1; then
    if [[ -n "${JAC_BIN:-}" && -d "$JAC_BIN" ]]; then
        export PATH="$JAC_BIN:$PATH"
    else
        echo "warning: no 'jac' on PATH -- every byLLM arm will fail." >&2
        echo "         Install the Jac toolchain or export JAC_BIN=<dir with jac>." >&2
    fi
fi
command -v jac >/dev/null 2>&1 && echo "jac: $(command -v jac)"

# ---------------------------------------------------------------------------
# 2. Static verification -- no model, no weights, no tokens
# ---------------------------------------------------------------------------
if [[ "$VERIFY_ONLY" -eq 1 ]]; then
    exec "$PY" -m bench.verify
fi

# ---------------------------------------------------------------------------
# 3. The shared model server, and a capability probe before anything expensive
# ---------------------------------------------------------------------------
echo
echo "--- model server ---"
SERVED="$("$PY" - <<'EOF'
import sys
from bench import services, config
services.ensure_ollama()
sys.stdout.write(services.ensure_model())
EOF
)"
export BENCH_SERVED_MODEL="$SERVED"

echo
if ! "$PY" -m bench.services --probe; then
    echo
    echo "The probe failed. Every benchmark here drives the model with tool calls;"
    echo "running the sweep anyway produces a table of zeros that looks like a"
    echo "framework result and is not one. Fix the model or endpoint first, or"
    echo "re-run with --probe-only to iterate."
    exit 1
fi
[[ "$PROBE_ONLY" -eq 1 ]] && exit 0

# ---------------------------------------------------------------------------
# 4. Clean, if asked
# ---------------------------------------------------------------------------
if [[ "$DO_CLEAN" -eq 1 ]]; then
    echo
    echo "--- clean ---"
    "$PY" -m bench.clean --yes
fi

# ---------------------------------------------------------------------------
# 5. The token ledger, then the sweep
# ---------------------------------------------------------------------------
RUN_ID="${BENCH_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
export BENCH_RUN_ID="$RUN_ID"

echo
echo "--- token ledger ---"
# Quoted delimiter: the body is Python, not shell. Unquoted, bash expands every
# $NAME inside it -- including ones that only appear in a comment -- and under
# `set -u` an unset one aborts the run. The run id travels in the environment
# instead, which is where config.ledger_path() reads it from anyway.
"$PY" - <<'PYLEDGER'
from bench import services, config

# config.ledger_path() is the single definition of where the ledger lives; every
# arm is handed it as PROXY_LOG, and RagGPT's scorer joins tokens from that file.
ledger = config.ledger_path()
ledger.parent.mkdir(parents=True, exist_ok=True)
services.write_pricing_table(config.RUNS_ROOT / "pricing.json")
if config.use_proxy():
    services.ensure_proxy(ledger)
else:
    print("proxy: disabled (BENCH_NO_PROXY)")
PYLEDGER

echo
"$PY" -m bench.run_all --run-id "$RUN_ID" "${SWEEP_ARGS[@]}" || SWEEP_FAILED=1

echo
echo "--- report ---"
"$PY" -m bench.report --run-id "$RUN_ID" || true

echo
echo "Run $RUN_ID complete. Summary: bench/runs/$RUN_ID/summary.md"
exit "${SWEEP_FAILED:-0}"
