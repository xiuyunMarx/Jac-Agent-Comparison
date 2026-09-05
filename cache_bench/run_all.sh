#!/usr/bin/env bash
# Drive every byLLM arm with invariant prompt hoisting off and on, logging each
# request, then score the logs. Usage:
#   OLLAMA_API_KEY=... cache_bench/run_all.sh [arm ...]      # arms: meeting email factcheck codeagent ytnav
# Env knobs: BENCH_MODEL (glm-5.2), OPENAI_BASE_URL (https://ollama.com/v1),
#            MODES ("baseline hoisted"), MEETING_CASES, EMAIL_BATCHES, CODE_INSTANCES,
#            FC_LIMIT (3 claims), YTNAV_LIMIT (5 questions), CODE_WORKERS (1; two jac workers race on the runtime Postgres), RESULTS (cache_bench/results), JAC_BIN.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${RESULTS:-$ROOT/cache_bench/results}"
MODES="${MODES:-baseline hoisted}"
ARMS=("$@"); [[ ${#ARMS[@]} -eq 0 ]] && ARMS=(meeting email factcheck codeagent ytnav)

: "${OLLAMA_API_KEY:?export OLLAMA_API_KEY (ollama.com key)}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$OLLAMA_API_KEY}"
export BENCH_MODEL="${BENCH_MODEL:-glm-5.2}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://ollama.com/v1}"
export OPENAI_API_BASE="$OPENAI_BASE_URL" OLLAMA_API_BASE="$OPENAI_BASE_URL"
export CODEAGENT_MODEL="${CODEAGENT_MODEL:-openai/$BENCH_MODEL}"
JAC_BIN="${JAC_BIN:-$HOME/miniconda3/envs/jac-main/bin}"; export PATH="$JAC_BIN:$PATH"
PY="${PY:-$HOME/miniconda3/envs/jaseci/bin/python}"        # drives meeting/email/codeagent evals
YTNAV_PY="${YTNAV_PY:-$HOME/miniconda3/envs/ytnav/bin/python}"
SCORE_PY="${SCORE_PY:-$ROOT/CodeAgent/Jac/.jac/venv/bin/python}"   # has litellm for the tokenizer
echo "jac: $(command -v jac) ($(jac --version 2>/dev/null | tail -1))  model: $CODEAGENT_MODEL  modes: $MODES  arms: ${ARMS[*]}"

stamp() { date '+%F %T'; }
NICE="${NICE:-nice -n 15}"          # stay out of the way of latency-sensitive work on the box
RETRIES="${RETRIES:-2}"             # re-run items the OOM killer (or a flaky call) took out
run_mode() {   # run_mode <mode> <arm> <logdir> <cmd...>   (appends to the logs; callers loop for retries)
    local mode="$1" arm="$2" dir="$3"; shift 3
    mkdir -p "$dir"
    local hoist=0; [[ "$mode" == "hoisted" ]] && hoist=1
    echo "[$(stamp)] $arm/$mode start: $*" | cut -c1-200
    BYLLM_INVARIANT_HOISTING=$hoist BYLLM_PROMPT_LOG="$dir/prompts.jsonl" BYLLM_USAGE_LOG="$dir/usage.jsonl" \
        $NICE "$@" >> "$dir/driver.log" 2>&1 || echo "[$(stamp)] $arm/$mode driver exited $? (see $dir/driver.log)"
    echo "[$(stamp)] $arm/$mode done: $(wc -l < "$dir/prompts.jsonl" 2>/dev/null || echo 0) calls logged so far"
}
meeting_failed() {   # cases whose result file is missing or not successful
    local c; for c in "$@"; do
        python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('success') else 1)" \
            "$ROOT/meeting-assistant/eval/runs/results_byLLM_${c}_r1.json" 2>/dev/null || echo "$c"
    done
}
email_failed() {     # batches with no results file (the driver's own success rule)
    local b; for b in "$@"; do
        [[ -f "$ROOT/Email-Auto-response/byLLM/mock_output/results_$b.json" ]] || echo "$b"
    done
}
fc_failed() {        # claims whose result json is missing or carries an error
    local out="$1"; shift; local i; for i in "$@"; do
        python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('error') is None else 1)" \
            "$out/jac_$(printf %03d "$i").json" 2>/dev/null || echo "$i"
    done
}
code_failed() {      # instances whose result.json is missing or carries an error
    local out="$1"; shift; local i; for i in "$@"; do
        python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if not d.get('error') else 1)" \
            "$out/logs/$i/result.json" 2>/dev/null || echo "$i"
    done
}

for arm in "${ARMS[@]}"; do
  for mode in $MODES; do
    dir="$RESULTS/$arm/$mode"
    case "$arm" in
      meeting)
        todo=(${MEETING_CASES:-$(cd "$ROOT/meeting-assistant/datasets" && ls meeting_*.txt | sed 's/\.txt$//' | tr '\n' ' ')})
        for ((try=0; try<=RETRIES && ${#todo[@]}>0; try++)); do
          run_mode "$mode" "$arm" "$dir" "$PY" "$ROOT/meeting-assistant/eval/run.py" --impl byLLM --cases "${todo[@]}" --repeat 1
          todo=($(meeting_failed "${todo[@]}")); [[ ${#todo[@]} -gt 0 ]] && echo "[$(stamp)] $arm/$mode retry ${todo[*]}"
        done ;;
      email)
        todo=(${EMAIL_BATCHES:-$(cd "$ROOT/Email-Auto-response/mock_mailbox/datasets" && ls batch_*.json | sed 's/\.json$//' | tr '\n' ' ')})
        rm -f "$ROOT"/Email-Auto-response/byLLM/mock_output/results_*.json   # success == file exists; never trust a stale one
        for ((try=0; try<=RETRIES && ${#todo[@]}>0; try++)); do
          run_mode "$mode" "$arm" "$dir" "$PY" "$ROOT/Email-Auto-response/eval/run.py" --impl byLLM --batches "${todo[@]}"
          todo=($(email_failed "${todo[@]}")); [[ ${#todo[@]} -gt 0 ]] && echo "[$(stamp)] $arm/$mode retry ${todo[*]}"
        done ;;
      factcheck)
        n="${FC_LIMIT:-3}"; force=--force      # first pass reruns everything; retries only redo failed claims
        for ((try=0; try<=RETRIES; try++)); do
          run_mode "$mode" "$arm" "$dir" "$PY" "$ROOT/FactCheck/eval/run.py" --arms jac --limit "$n" --serial --out "$dir/out" $force
          force=""; todo=($(fc_failed "$dir/out" $(seq 1 "$n"))); [[ ${#todo[@]} -eq 0 ]] && break
          echo "[$(stamp)] $arm/$mode retry ${todo[*]}"
        done ;;
      codeagent)
        todo=(${CODE_INSTANCES:-django__django-11099 pytest-dev__pytest-5227 sympy__sympy-13480 astropy__astropy-14995})
        for ((try=0; try<=RETRIES && ${#todo[@]}>0; try++)); do
          run_mode "$mode" "$arm" "$dir" "$PY" "$ROOT/CodeAgent/swebench_bridge/run_agent.py" --framework jac \
            --instance-ids "${todo[@]}" --run-id "cache-$mode-jac" --output-dir "$dir/swebench" \
            --runtime docker --workers "${CODE_WORKERS:-1}" --model "$CODEAGENT_MODEL" --jac "$JAC_BIN/jac" --force
          todo=($(code_failed "$dir/swebench/cache-$mode-jac" "${todo[@]}")); [[ ${#todo[@]} -gt 0 ]] && echo "[$(stamp)] $arm/$mode retry ${todo[*]}"
        done ;;
      ytnav)
        mkdir -p "$dir"; head -n "${YTNAV_LIMIT:-5}" "$ROOT/YTNavigator/datasets/questions.jsonl" > "$dir/questions.jsonl"
        run_mode "$mode" "$arm" "$dir" "$YTNAV_PY" "$ROOT/YTNavigator/eval/e2e.py" --impl byllm --langgraph-python "$YTNAV_PY" \
            --questions "$dir/questions.jsonl" ;;
      *) echo "unknown arm $arm" >&2; exit 2 ;;
    esac
  done
  # score this arm: one label per mode, one session per log file
  specs=()
  for mode in $MODES; do
    dir="$RESULTS/$arm/$mode"
    if [[ "$arm" == codeagent ]]; then
      for f in "$dir"/swebench/cache-$mode-jac/logs/*/byllm_prompts.jsonl; do [[ -s "$f" ]] && specs+=("$mode=$f"); done
    else
      [[ -s "$dir/prompts.jsonl" ]] && specs+=("$mode=$dir/prompts.jsonl")
    fi
  done
  shared=(); [[ "$arm" == codeagent ]] && shared=(--shared)   # instances run against one server, one radix tree
  if [[ ${#specs[@]} -gt 0 ]]; then
    "$SCORE_PY" "$ROOT/cache_bench/score.py" "${specs[@]}" "${shared[@]}" --per-scope 2>/dev/null | tee "$RESULTS/$arm/summary.txt"
    "$SCORE_PY" "$ROOT/cache_bench/score.py" "${specs[@]}" "${shared[@]}" --json 2>/dev/null > "$RESULTS/$arm/summary.json"
  fi
done
echo "[$(stamp)] all done -> $RESULTS"
