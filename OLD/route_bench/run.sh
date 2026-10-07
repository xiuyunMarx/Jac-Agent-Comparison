#!/usr/bin/env bash
# Learnt model routing on the Email byLLM arm: strong-only vs cheap-only vs routed.
#   set -a; . ./.env; set +a
#   route_bench/run.sh collect          # strong model on TRAIN_BATCHES, prompt log -> training data
#   route_bench/run.sh train            # replay both candidates, train LLMRouter KNN (llmrouter venv)
#   route_bench/run.sh eval [mode ...]  # modes: strong cheap routed   (on EVAL_BATCHES)
#   route_bench/run.sh report
# Env knobs: STRONG (glm-5.2), CHEAP (gemma4:31b), TRAIN_BATCHES, EVAL_BATCHES, PORT (8765), RESULTS.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RB="$ROOT/route_bench"
RESULTS="$(mkdir -p "${RESULTS:-$RB/results}" && cd "${RESULTS:-$RB/results}" && pwd)"   # absolute: the jac subprocess runs from the agent dir
STRONG="${STRONG:-glm-5.2}"
CHEAP="${CHEAP:-gemma4:31b}"
TRAIN_BATCHES="${TRAIN_BATCHES:-batch_001 batch_002 batch_003}"
EVAL_BATCHES="${EVAL_BATCHES:-batch_004 batch_005 batch_006}"
PORT="${PORT:-8765}"
TRAIN_PROMPTS="$(realpath -m ${TRAIN_PROMPTS:-$RESULTS/collect/prompts.jsonl})"
ROUTER_DIR="$(realpath -m "${ROUTER_DIR:-$RESULTS/router}")"
EVAL_DIR="$(realpath -m "${EVAL_DIR:-$RESULTS/eval}")"
: "${OLLAMA_API_KEY:?export OLLAMA_API_KEY (ollama.com key)}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$OLLAMA_API_KEY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://ollama.com/v1}"
export OPENAI_API_BASE="$OPENAI_BASE_URL"
export BYLLM_INVARIANT_HOISTING=1            # same prompt shape in every mode
export EVAL_PRICING_FILE="$RB/pricing.json"
JAC_BIN="${JAC_BIN:-$HOME/miniconda3/envs/jac-main/bin}"; export PATH="$JAC_BIN:$PATH"
PY="${PY:-$HOME/miniconda3/envs/jaseci/bin/python}"          # drives Email-Auto-response/eval/run.py
RPY="${RPY:-$HOME/.venvs/llmrouter/bin/python}"                # LLMRouter + torch (CPU)
EMAIL="$ROOT/Email-Auto-response"
stamp() { date '+%F %T'; }

run_email() {   # run_email <dir> <batches...>  (env BENCH_MODEL / BYLLM_ROUTING* set by caller)
    local dir="$1"; shift
    mkdir -p "$dir"
    rm -f "$EMAIL"/byLLM/mock_output/results_*.json
    echo "[$(stamp)] email run -> $dir  model=$BENCH_MODEL routing=${BYLLM_ROUTING:-0} batches: $*"
    BYLLM_PROMPT_LOG="$dir/prompts.jsonl" BYLLM_USAGE_LOG="$dir/usage.jsonl" BYLLM_ROUTE_LOG="$dir/routes.jsonl" \
        "$PY" "$EMAIL/eval/run.py" --impl byLLM --no-score --batches "$@" > "$dir/driver.log" 2>&1 \
        || echo "[$(stamp)] driver exited $? (see $dir/driver.log)"
    mkdir -p "$dir/mock_output"; cp "$EMAIL"/byLLM/mock_output/results_*.json "$dir/mock_output/" 2>/dev/null || true
    echo "[$(stamp)] done: $(ls "$dir"/mock_output | wc -l) result files, $(wc -l < "$dir/usage.jsonl" 2>/dev/null || echo 0) LLM calls"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  collect)
    BENCH_MODEL="$STRONG" BYLLM_ROUTING=0 run_email "$RESULTS/collect" $TRAIN_BATCHES ;;
  train)
    LLMROUTER_EMBEDDING_DEVICE=cpu CUDA_VISIBLE_DEVICES= "$RPY" "$RB/build_router.py" \
        --prompts $TRAIN_PROMPTS --out "$ROUTER_DIR" \
        --candidates "openai/$CHEAP" "openai/$STRONG" --reference "openai/$STRONG" "$@" ;;
  serve)   # foreground router service (eval starts its own)
    LLMROUTER_EMBEDDING_DEVICE=cpu CUDA_VISIBLE_DEVICES= exec "$RPY" "$RB/router_server.py" --policy knn \
        --config "$ROUTER_DIR/knnrouter_test.yaml" --port "$PORT" "$@" ;;
  eval)
    modes=("$@"); [[ ${#modes[@]} -eq 0 ]] && modes=(strong cheap routed)
    for mode in "${modes[@]}"; do
      case "$mode" in
        strong) BENCH_MODEL="$STRONG" BYLLM_ROUTING=0 run_email "$EVAL_DIR/strong" $EVAL_BATCHES ;;
        cheap)  BENCH_MODEL="$CHEAP"  BYLLM_ROUTING=0 run_email "$EVAL_DIR/cheap" $EVAL_BATCHES ;;
        routed)
          mkdir -p "$EVAL_DIR/routed"
          LLMROUTER_EMBEDDING_DEVICE=cpu CUDA_VISIBLE_DEVICES= "$RPY" "$RB/router_server.py" --policy knn \
              --config "$ROUTER_DIR/knnrouter_test.yaml" --port "$PORT" --log "$EVAL_DIR/routed/router.jsonl" \
              > "$EVAL_DIR/routed/router_server.log" 2>&1 &
          spid=$!
          for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/" >/dev/null 2>&1 && break; sleep 2; done
          curl -sf "http://127.0.0.1:$PORT/" >/dev/null || { echo "router service did not come up"; kill $spid; exit 1; }
          BENCH_MODEL="$STRONG" BYLLM_ROUTING=1 BYLLM_ROUTING_ENDPOINT="http://127.0.0.1:$PORT/route" \
              BYLLM_ROUTING_CANDIDATES="openai/$CHEAP,openai/$STRONG" run_email "$EVAL_DIR/routed" $EVAL_BATCHES
          kill $spid 2>/dev/null || true ;;
        *) echo "unknown mode $mode"; exit 2 ;;
      esac
    done ;;
  judge)   # LLM-judge the drafts of each mode (glm-5.2 via ollama.com as judge), labels = mode names
    for mode in strong cheap routed; do
      d="$EVAL_DIR/$mode"; [[ -d "$d/mock_output" ]] || continue
      EVAL_JUDGE_MODEL="${JUDGE_MODEL:-$STRONG}" "$PY" "$EMAIL/eval/score.py" "$d/mock_output" --judge --label "$(basename "$EVAL_DIR")_$mode" \
          > "$d/judge.log" 2>&1 || echo "judge failed for $mode (see $d/judge.log)"
      cp "$EMAIL"/eval/out/scores_"$(basename "$EVAL_DIR")_$mode"_*.json "$d/" 2>/dev/null || true
      echo "[$(stamp)] judged $mode"
    done ;;
  report)
    "$PY" "$RB/report.py" --results "$RESULTS" --eval "$EVAL_DIR" --router "$ROUTER_DIR" "$@" ;;
  *) sed -n 2,8p "$0"; exit 2 ;;
esac
