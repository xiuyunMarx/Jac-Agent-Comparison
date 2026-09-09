#!/usr/bin/env bash
# Learnt routing on the other byLLM arms, a few cases each.
#   set -a; . ./.env; set +a
#   route_bench/run_arm.sh <arm> collect|train|eval [modes]|score|all      arms: meeting factcheck ytnav codeagent
# Train cases -> collect (strong model, logs every request) -> train (replay both candidates, KNN) -> eval cases
# on strong / cheap / routed -> score with the arm's own scorer. Results: route_bench/results/<arm>/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RB="$ROOT/route_bench"
ARM="${1:?arm}"; STEP="${2:?step}"; shift 2 || true
RES="$RB/results/${RESULTS_NAME:-$ARM}"; mkdir -p "$RES"   # RESULTS_NAME: alternate results dir for the same arm
STRONG="${STRONG:-glm-5.2}"; CHEAP="${CHEAP:-gemma4:31b}"
: "${OLLAMA_API_KEY:?export OLLAMA_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$OLLAMA_API_KEY}" OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://ollama.com/v1}"
export OPENAI_API_BASE="$OPENAI_BASE_URL" BYLLM_INVARIANT_HOISTING=1
JAC_BIN="${JAC_BIN:-$HOME/miniconda3/envs/jac-main/bin}"; export PATH="$JAC_BIN:$PATH"
PY="${PY:-$HOME/miniconda3/envs/jaseci/bin/python}"
YTNAV_PY="${YTNAV_PY:-$HOME/miniconda3/envs/ytnav/bin/python}"
RPY="${RPY:-$HOME/.venvs/llmrouter/bin/python}"
NICE="${NICE:-nice -n 10}"
stamp() { date '+%F %T'; }

case "$ARM" in
  meeting)   PORT="${PORT:-8771}"; TRAIN="${TRAIN:-meeting_001 meeting_002 meeting_003 meeting_004}"; EVAL="${EVAL:-meeting_005 meeting_006 meeting_007 meeting_008}" ;;
  factcheck) PORT="${PORT:-8772}"; TRAIN="${TRAIN:-1 2 3}"; EVAL="${EVAL:-4 5 6}" ;;
  ytnav)     PORT="${PORT:-8773}"; TRAIN="${TRAIN:-1 2 3 4 5 6}"; EVAL="${EVAL:-7 8 9 10 11 12}" ;;
  codeagent) PORT="${PORT:-8774}"; TRAIN="${TRAIN:-django__django-11099 sympy__sympy-13480}"; EVAL="${EVAL:-pytest-dev__pytest-5227 astropy__astropy-14995}" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

# run_cases <dir> <model> <cases...>   (routing env set by the caller)
run_cases() {
  local dir="$1" model="$2"; shift 2; mkdir -p "$dir"
  echo "[$(stamp)] $ARM -> $dir model=$model routing=${BYLLM_ROUTING:-0} cases: $*"
  export BYLLM_PROMPT_LOG="$dir/prompts.jsonl" BYLLM_USAGE_LOG="$dir/usage.jsonl" BYLLM_ROUTE_LOG="$dir/routes.jsonl"
  case "$ARM" in
    meeting)
      BENCH_MODEL="$model" $NICE "$PY" "$ROOT/meeting-assistant/eval/run.py" --impl byLLM --cases "$@" --repeat 1 > "$dir/driver.log" 2>&1 || echo "driver exited $?"
      mkdir -p "$dir/runs"; for c in "$@"; do cp "$ROOT/meeting-assistant/eval/runs/results_byLLM_${c}_r1.json" "$dir/runs/" 2>/dev/null || echo "missing result $c"; done ;;
    factcheck)
      { grep "^#" "$ROOT/FactCheck/hover_claims.tsv" | head -1; grep -v "^#" "$ROOT/FactCheck/hover_claims.tsv" | grep -v "^\s*$" | sed -n "$(printf '%sp;' "$@")"; } > "$dir/claims.tsv"
      FC_MODEL="openai/$model" $NICE "$PY" "$ROOT/FactCheck/eval/run.py" --arms jac --claims "$dir/claims.tsv" --serial --out "$dir/out" --force > "$dir/driver.log" 2>&1 || echo "driver exited $?" ;;
    ytnav)
      sed -n "$(printf '%sp;' "$@")" "$ROOT/YTNavigator/datasets/questions.jsonl" > "$dir/questions.jsonl"
      BENCH_MODEL="$model" $NICE "$YTNAV_PY" "$ROOT/YTNavigator/eval/e2e.py" --impl byllm --langgraph-python "$YTNAV_PY" --questions "$dir/questions.jsonl" > "$dir/driver.log" 2>&1 || echo "driver exited $?"
      cp "$ROOT/YTNavigator/eval/out/results_byllm.jsonl" "$ROOT/YTNavigator/eval/out/report.json" "$dir/" 2>/dev/null || echo "missing ytnav results" ;;
    codeagent)
      local runid="route-$(basename "$dir")-jac"
      $NICE "$PY" "$ROOT/CodeAgent/swebench_bridge/run_agent.py" --framework jac --instance-ids "$@" --run-id "$runid" \
        --output-dir "$dir/swebench" --runtime docker --workers 1 --model "openai/$model" --jac "$JAC_BIN/jac" --force > "$dir/driver.log" 2>&1 || echo "driver exited $?" ;;
  esac
  echo "[$(stamp)] $ARM done: $(cat "$dir"/usage.jsonl "$dir"/out/jac_*.calls.jsonl "$dir"/swebench/*/logs/*/byllm_usage.jsonl 2>/dev/null | wc -l) LLM calls logged"
}

prompt_logs() {   # every prompt log of a run dir
  ls "$1"/prompts.jsonl "$1"/swebench/*/logs/*/byllm_prompts.jsonl 2>/dev/null || true
}

do_collect() { BYLLM_ROUTING=0 run_cases "$RES/collect" "$STRONG" $TRAIN; }
do_train() {
  LLMROUTER_EMBEDDING_DEVICE=cpu CUDA_VISIBLE_DEVICES= "$RPY" "$RB/build_router.py" --prompts ${TRAIN_PROMPTS:-$(prompt_logs "$RES/collect")} \
    --out "$RES/router" --candidates "openai/$CHEAP" "openai/$STRONG" --reference "openai/$STRONG" --workers 3 "$@" 2>&1 | grep -vE "^Successfully loaded|^\s*$" | tail -60 | tee "$RES/train.log"
}
do_eval() {
  local modes=("$@"); [[ ${#modes[@]} -eq 0 ]] && modes=(strong cheap routed)
  for mode in "${modes[@]}"; do
    case "$mode" in
      strong) BYLLM_ROUTING=0 run_cases "$RES/eval/strong" "$STRONG" $EVAL ;;
      cheap)  BYLLM_ROUTING=0 run_cases "$RES/eval/cheap" "$CHEAP" $EVAL ;;
      routed)
        mkdir -p "$RES/eval/routed"
        LLMROUTER_EMBEDDING_DEVICE=cpu CUDA_VISIBLE_DEVICES= "$RPY" "$RB/router_server.py" --policy knn --config "$RES/router/knnrouter_test.yaml" \
          --port "$PORT" --log "$RES/eval/routed/router.jsonl" > "$RES/eval/routed/router_server.log" 2>&1 &
        local spid=$!
        for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/" >/dev/null 2>&1 && break; sleep 2; done
        curl -sf "http://127.0.0.1:$PORT/" >/dev/null || { echo "router service did not come up"; kill $spid; exit 1; }
        BYLLM_ROUTING=1 BYLLM_ROUTING_ENDPOINT="http://127.0.0.1:$PORT/route" BYLLM_ROUTING_CANDIDATES="openai/$CHEAP,openai/$STRONG" \
          run_cases "$RES/eval/routed" "$STRONG" $EVAL
        kill $spid 2>/dev/null || true ;;
    esac
  done
}
do_score() {
  for mode in strong cheap routed; do
    local d="$RES/eval/$mode"; [[ -d "$d" ]] || continue
    case "$ARM" in
      meeting)   "$PY" "$ROOT/meeting-assistant/eval/score.py" "$d"/runs/*.json "$@" > "$d/score.txt" 2>&1 || true; cp "$ROOT/meeting-assistant/eval/out/summary.json" "$d/scores.json" 2>/dev/null || true ;;
      factcheck) "$PY" "$ROOT/FactCheck/eval/score.py" --out "$d/out" --report "$d/report.md" > "$d/score.txt" 2>&1 || true ;;
      ytnav)     "$YTNAV_PY" "$ROOT/YTNavigator/eval/score.py" "$d/results_byllm.jsonl" --questions "$d/questions.jsonl" --report "$d/report.json" "$@" > "$d/score.txt" 2>&1 || true ;;
      codeagent) for p in "$d"/swebench/*/predictions.jsonl; do "$PY" "$ROOT/CodeAgent/swebench_bridge/grade.py" --predictions "$p" --runtime docker --workers 1 > "$d/grade.log" 2>&1 || echo "grade failed $mode"; done ;;
    esac
    echo "[$(stamp)] scored $ARM/$mode"
  done
}

case "$STEP" in
  collect) do_collect ;;
  train)   do_train "$@" ;;
  eval)    do_eval "$@" ;;
  score)   do_score "$@" ;;
  all)     do_collect; do_train; do_eval; do_score ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
