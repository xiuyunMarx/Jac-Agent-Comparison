#!/usr/bin/env bash
# Sweep every agent x every implementation against one base model.
#
# Default model: Claude Sonnet
#
#   ./sweep.sh                        # all five agents, every arm
#   ./sweep.sh factcheck meeting      # only these agents
#   ./sweep.sh --check                # preflight only: key, endpoint, venvs, docker
#   SWEEP_PROVIDER=openai ./sweep.sh  # same sweep on an OpenAI key ($OPENAI_API_KEY)
#   SMOKE=1 ./sweep.sh                # 1-2 items per dataset, to prove the wiring
#   DRY_RUN=1 ./sweep.sh              # print every command, call nothing
#   MEETING_ARMS="byLLM CrewAI" ./sweep.sh meeting
#
# Knobs: SWEEP_PROVIDER (anthropic|openai|ollama) SWEEP_MODEL SWEEP_BASE_URL
#        RESULTS RETRIES JUDGE NICE
#   arms:        FACTCHECK_ARMS EMAIL_ARMS MEETING_ARMS YTNAV_ARMS CODEAGENT_ARMS
#   datasets:    FC_LIMIT EMAIL_BATCHES MEETING_CASES MEETING_REPEAT YTNAV_LIMIT
#                CODE_INSTANCES CODE_MAX_STEPS CODE_WORKERS CODE_RUNTIME
#   interpreters: PY YTNAV_PY JAC_BIN CREW_PYTHON SCORE_PY
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

case "${1:-}" in -h|--help) sed -n '2,19p' "$0"; exit 0 ;; esac

# ---------------------------------------------------------------- model + auth
[[ -f "$ROOT/.env" ]] && { set -a; . "$ROOT/.env"; set +a; }

# One provider for the whole sweep, chosen explicitly -- never inferred from
# whichever key happens to be in the environment, because a silent provider
# switch mid-benchmark is indistinguishable from a framework effect.
SWEEP_PROVIDER="${SWEEP_PROVIDER:-anthropic}"
case "$SWEEP_PROVIDER" in
    anthropic) _model=claude-sonnet-5; _base=https://api.anthropic.com/v1; _keyvar=ANTHROPIC_API_KEY ;;
    openai)    _model=gpt-5;           _base=https://api.openai.com/v1;    _keyvar=OPENAI_API_KEY ;;
    ollama)    _model=glm-5.2;         _base=https://ollama.com/v1;        _keyvar=OLLAMA_API_KEY ;;
    *) echo "unknown SWEEP_PROVIDER '$SWEEP_PROVIDER' (anthropic openai ollama)" >&2; exit 2 ;;
esac
SWEEP_MODEL="${SWEEP_MODEL:-$_model}"                  # bare id; arms add the provider prefix
SWEEP_BASE_URL="${SWEEP_BASE_URL:-$_base}"             # no trailing slash
KEY="${!_keyvar:-}"

if [[ -z "$KEY" ]]; then
    # CLAUDE_CODE_OAUTH_TOKEN authenticates the Claude Code client, not the
    # Messages API or its OpenAI-compatible front end. Try it only on request.
    if [[ "$SWEEP_PROVIDER" == anthropic && -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" && -n "${SWEEP_ALLOW_OAUTH:-}" ]]; then
        KEY="$CLAUDE_CODE_OAUTH_TOKEN"
        echo "warning: using CLAUDE_CODE_OAUTH_TOKEN as an API key; expect 401 unless a proxy accepts it" >&2
    else
        echo "$_keyvar is not set (SWEEP_PROVIDER=$SWEEP_PROVIDER)." >&2
        if [[ "$SWEEP_PROVIDER" == anthropic && -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]; then
            echo "  .env has CLAUDE_CODE_OAUTH_TOKEN, which is a Claude Code credential, not an API key." >&2
            echo "  For an OpenAI key instead: SWEEP_PROVIDER=openai ./sweep.sh" >&2
        fi
        exit 2
    fi
fi

# One model, one endpoint, one key, spelled the way each arm expects to read it.
export BENCH_MODEL="$SWEEP_MODEL"                  # email / meeting / ytnav (bare id)
export FC_MODEL="openai/$SWEEP_MODEL"              # FactCheck LangGraph + OpenaiSDK
export FC_MODEL_BASE="$SWEEP_BASE_URL"
export CODEAGENT_MODEL="openai/$SWEEP_MODEL"       # CodeAgent, all four arms
export OPENAI_BASE_URL="$SWEEP_BASE_URL"
export OPENAI_API_BASE="$SWEEP_BASE_URL"
export OLLAMA_API_BASE="$SWEEP_BASE_URL"           # CodeAgent/Jac reads this name
export OPENAI_API_KEY="$KEY"
export ANTHROPIC_API_KEY="$KEY"
export OLLAMA_API_KEY="$KEY"                       # FactCheck/Jac passes this as its api_key
# litellm knows no context length for "openai/claude-*", and byLLM's
# auto-compaction misreads a zero window; state it rather than let it guess.
export BENCH_CTX="${BENCH_CTX:-200000}"
export PYTHONUNBUFFERED=1

# --------------------------------------------------------------- interpreters
PY="${PY:-$HOME/miniconda3/envs/jaseci/bin/python}"          # factcheck / email / meeting / codeagent drivers
YTNAV_PY="${YTNAV_PY:-$HOME/miniconda3/envs/ytnav/bin/python}"
JAC_BIN="${JAC_BIN:-$HOME/miniconda3/envs/jac-main/bin}"
SCORE_PY="${SCORE_PY:-$ROOT/CodeAgent/Jac/.jac/venv/bin/python}"   # has litellm, for judged scoring
export JAC_BIN PATH="$JAC_BIN:$PATH"

# --------------------------------------------------------------------- arms
FACTCHECK_ARMS="${FACTCHECK_ARMS:-jac langgraph openai}"
EMAIL_ARMS="${EMAIL_ARMS:-byLLM CrewAI-LangGraph openai_sdk}"
MEETING_ARMS="${MEETING_ARMS:-byLLM CrewAI openai_sdk}"
YTNAV_ARMS="${YTNAV_ARMS:-byllm langgraph openai_sdk}"
CODEAGENT_ARMS="${CODEAGENT_ARMS:-jac langgraph openai nooa}"

# ------------------------------------------------------------------ datasets
if [[ -n "${SMOKE:-}" ]]; then
    FC_LIMIT="${FC_LIMIT:-2}"
    EMAIL_BATCHES="${EMAIL_BATCHES:-batch_001}"
    MEETING_CASES="${MEETING_CASES:-meeting_001}"
    YTNAV_LIMIT="${YTNAV_LIMIT:-3}"
    CODE_INSTANCES="${CODE_INSTANCES:-django__django-11099}"
    CODE_MAX_STEPS="${CODE_MAX_STEPS:-4}"
fi
FC_LIMIT="${FC_LIMIT:-0}"                                    # 0 = every claim
EMAIL_BATCHES="${EMAIL_BATCHES:-batch_001 batch_002 batch_003 batch_004 batch_005 batch_006}"
MEETING_CASES="${MEETING_CASES:-}"                           # empty = every case
MEETING_REPEAT="${MEETING_REPEAT:-1}"
YTNAV_LIMIT="${YTNAV_LIMIT:-0}"                              # 0 = every question
CODE_INSTANCES="${CODE_INSTANCES:-django__django-11099 pytest-dev__pytest-5227 sympy__sympy-13480 astropy__astropy-14995}"
CODE_MAX_STEPS="${CODE_MAX_STEPS:-10}"
CODE_WORKERS="${CODE_WORKERS:-1}"                            # two jac workers race on the runtime Postgres
CODE_RUNTIME="${CODE_RUNTIME:-docker}"

# -------------------------------------------------------------------- driving
RUN_ID="${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"
RESULTS="${RESULTS:-$ROOT/sweep_results}"
RUN_DIR="$RESULTS/$RUN_ID"
RETRIES="${RETRIES:-1}"          # extra attempts per arm; the runners resume rather than redo
JUDGE="${JUDGE:-}"               # non-empty adds LLM-as-judge scoring where the scorer supports it
NICE="${NICE:-nice -n 15}"
DRY_RUN="${DRY_RUN:-}"
CHECK_ONLY=""

APPS=()
for a in "$@"; do
    case "$a" in
        --check) CHECK_ONLY=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        -*) echo "unknown option $a" >&2; exit 2 ;;
        *) APPS+=("$a") ;;
    esac
done
[[ ${#APPS[@]} -eq 0 ]] && APPS=(factcheck email meeting ytnav codeagent)

stamp() { date '+%F %T'; }
log()   { echo "[$(stamp)] $*" | tee -a "$RUN_DIR/sweep.log"; }
record() {   # record <app> <arm> <status> <seconds> <logdir>
    printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >> "$RUN_DIR/summary.tsv"
}

run_step() {   # run_step <app> <arm> <logdir> <cmd...>
    local app="$1" arm="$2" dir="$3"; shift 3
    mkdir -p "$dir"
    if [[ -n "$DRY_RUN" ]]; then
        printf '  would run: %s\n' "$*" | tee -a "$RUN_DIR/sweep.log"
        record "$app" "$arm" dry 0 "$dir"
        return 0
    fi
    log "$app/$arm start"
    local t0=$SECONDS rc=0 try
    for ((try = 0; try <= RETRIES; try++)); do
        [[ $try -gt 0 ]] && log "$app/$arm attempt $((try + 1))"
        set +e
        $NICE "$@" >> "$dir/driver.log" 2>&1
        rc=$?
        set -e
        [[ $rc -eq 0 ]] && break
    done
    local secs=$((SECONDS - t0))
    if [[ $rc -eq 0 ]]; then
        log "$app/$arm ok in ${secs}s"
        record "$app" "$arm" ok "$secs" "$dir"
    else
        log "$app/$arm FAILED (exit $rc) after ${secs}s -- see $dir/driver.log"
        record "$app" "$arm" "failed:$rc" "$secs" "$dir"
    fi
    return 0                      # one broken arm must not end the sweep
}

stash_prior() {   # stash_prior <app> <path...>   move an earlier model's results out of the way
    # Both the email and meeting scorers read a whole directory, and the email
    # runner's success rule is "the file is there". Left in place, a previous
    # model's results would be scored alongside this run's and would mask a
    # crash as a success. They are moved, never deleted: they are results.
    # Mirror each file's path relative to the repo, rather than guessing how
    # many levels up the arm name sits. All three email arms write a directory
    # called mock_output holding results_batch_00N.json, so anything less than
    # the full relative path silently overwrites two arms out of three.
    local app="$1"; shift
    local dest="$RESULTS/_prior/$RUN_ID/$app" moved=0 p rel
    for p in "$@"; do
        [[ -e "$p" ]] || continue
        rel="${p#"$ROOT"/}"
        mkdir -p "$dest/$(dirname "$rel")"
        mv "$p" "$dest/$rel" 2>/dev/null && moved=$((moved + 1))
    done
    [[ $moved -gt 0 ]] && log "$app: stashed $moved prior result file(s) -> $dest"
    true
}

archive() {   # archive <destdir> <path...>   copy whatever an app wrote where it writes it
    # Mirror the repo-relative path, for the same reason stash_prior does:
    # three arms each own a directory called mock_output.
    local dest="$1"; shift
    mkdir -p "$dest"
    local p rel
    for p in "$@"; do
        [[ -e "$p" ]] || continue
        rel="${p#"$ROOT"/}"
        mkdir -p "$dest/$(dirname "$rel")"
        cp -r "$p" "$dest/$rel" 2>/dev/null || true
    done
    true
}

# -------------------------------------------------------------------- preflight
preflight() {
    local bad=0 p
    echo "provider: $SWEEP_PROVIDER   model: $SWEEP_MODEL   endpoint: $SWEEP_BASE_URL"
    echo "apps:     ${APPS[*]}"
    echo "run dir:  $RUN_DIR"
    echo
    for p in "$PY" "$YTNAV_PY" "$JAC_BIN/jac"; do
        [[ -x "$p" ]] && echo "  ok       $p" || { echo "  MISSING  $p"; bad=1; }
    done
    for p in "$ROOT/Email-Auto-response/CrewAI-LangGraph/.venv/bin/python" \
             "$ROOT/meeting-assistant/CrewAI/.venv/bin/python"; do
        [[ -x "$p" ]] && echo "  ok       $p" || echo "  missing  $p (CrewAI arms will be skipped by their runner)"
    done
    if command -v docker > /dev/null && docker ps > /dev/null 2>&1; then
        docker ps --format '{{.Names}}' | grep -qx ytnav-bench-pg \
            && echo "  ok       docker, ytnav-bench-pg up" \
            || echo "  ok       docker (ytnav-bench-pg down; eval/e2e.py starts it)"
    else
        echo "  MISSING  docker -- ytnav and codeagent need it"; bad=1
    fi
    # FactCheck's Jac arm is the one place the endpoint is not an env knob.
    grep -q 'glob MODEL_BASE: str = "https' "$ROOT/FactCheck/Jac/fact_check.jac" 2>/dev/null && {
        echo "  WARN     FactCheck/Jac/fact_check.jac hardcodes MODEL_BASE to ollama.com and passes it"
        echo "           as config[\"base_url\"]; that arm may ignore \$SWEEP_BASE_URL. Make it read"
        echo "           FC_MODEL_BASE, like the LangGraph and OpenaiSDK arms already do."
    }
    echo
    if [[ -n "$DRY_RUN" ]]; then
        echo "  skipped  endpoint probe (dry run)"
    else
        # GET the model rather than completing with it: this separates "the key
        # works and can see this model" from "the arms send parameters this
        # model accepts", which are different failures with different fixes.
        local code
        code=$(curl -sS -o /dev/null -w '%{http_code}' -m 30 \
            -H "Authorization: Bearer $KEY" \
            "$SWEEP_BASE_URL/models/$SWEEP_MODEL" 2>/dev/null || echo 000)
        case "$code" in
            200) echo "  ok       $SWEEP_BASE_URL can see $SWEEP_MODEL (HTTP 200)" ;;
            404|405) echo "  WARN     $SWEEP_BASE_URL/models/$SWEEP_MODEL -> HTTP $code"
                     echo "           either this key cannot see that model, or the endpoint has no /models" ;;
            *) echo "  FAIL     $SWEEP_BASE_URL/models/$SWEEP_MODEL -> HTTP $code"; bad=1 ;;
        esac
    fi
    return $bad
}

# ------------------------------------------------------------------------ apps
sweep_factcheck() {
    local out="$RUN_DIR/factcheck/out" arm limit=()
    [[ "$FC_LIMIT" != 0 ]] && limit=(--limit "$FC_LIMIT")
    for arm in $FACTCHECK_ARMS; do
        run_step factcheck "$arm" "$RUN_DIR/factcheck/$arm" \
            "$PY" "$ROOT/FactCheck/eval/run.py" --arms "$arm" --serial --out "$out" "${limit[@]}"
    done
    [[ -n "$DRY_RUN" ]] && return 0
    log "factcheck scoring"
    "$PY" "$ROOT/FactCheck/eval/score.py" --out "$out" \
        > "$RUN_DIR/factcheck/report.txt" 2>&1 || log "factcheck scoring failed"
}

sweep_email() {
    local arm dirs=()
    [[ -z "$DRY_RUN" ]] && stash_prior email \
        "$ROOT"/Email-Auto-response/*/mock_output/results_*.json \
        "$ROOT"/Email-Auto-response/eval/out/scores_byLLM_*.json \
        "$ROOT"/Email-Auto-response/eval/out/scores_CrewAI-LangGraph_*.json \
        "$ROOT"/Email-Auto-response/eval/out/scores_openai_sdk_*.json \
        "$ROOT/Email-Auto-response/eval/out/summary.json"
    for arm in $EMAIL_ARMS; do
        run_step email "$arm" "$RUN_DIR/email/$arm" \
            "$PY" "$ROOT/Email-Auto-response/eval/run.py" --impl "$arm" \
            --batches $EMAIL_BATCHES --no-score
        dirs+=("$ROOT/Email-Auto-response/$arm/mock_output")
    done
    [[ -n "$DRY_RUN" ]] && return 0
    # One scoring pass over every arm: score.py rebuilds summary.json from just
    # the directories it is handed, so per-arm scoring would leave a summary
    # describing whichever arm went last.
    log "email scoring"
    "$PY" "$ROOT/Email-Auto-response/eval/score.py" "${dirs[@]}" ${JUDGE:+--judge} \
        > "$RUN_DIR/email/report.txt" 2>&1 || log "email scoring failed"
    archive "$RUN_DIR/email/results" "${dirs[@]}"
}

sweep_meeting() {
    local arm cases=()
    [[ -z "$DRY_RUN" ]] && stash_prior meeting \
        "$ROOT"/meeting-assistant/eval/runs/results_*.json
    [[ -n "$MEETING_CASES" ]] && cases=(--cases $MEETING_CASES)
    for arm in $MEETING_ARMS; do
        run_step meeting "$arm" "$RUN_DIR/meeting/$arm" \
            "$PY" "$ROOT/meeting-assistant/eval/run.py" --impl "$arm" \
            --repeat "$MEETING_REPEAT" "${cases[@]}"
    done
    [[ -n "$DRY_RUN" ]] && return 0
    log "meeting scoring"
    "${SCORE_PY:-$PY}" "$ROOT/meeting-assistant/eval/score.py" \
        "$ROOT/meeting-assistant/eval/runs" ${JUDGE:+--judge} \
        > "$RUN_DIR/meeting/report.txt" 2>&1 || log "meeting scoring failed"
    archive "$RUN_DIR/meeting/results" "$ROOT/meeting-assistant/eval/runs"
}

sweep_ytnav() {
    local questions="$ROOT/YTNavigator/datasets/questions.jsonl" arm impl
    [[ -z "$DRY_RUN" ]] && stash_prior ytnav \
        "$ROOT"/YTNavigator/eval/out/results_*.jsonl "$ROOT/YTNavigator/eval/out/report.json"
    if [[ "$YTNAV_LIMIT" != 0 ]]; then
        mkdir -p "$RUN_DIR/ytnav"
        head -n "$YTNAV_LIMIT" "$questions" > "$RUN_DIR/ytnav/questions.jsonl"
        questions="$RUN_DIR/ytnav/questions.jsonl"
    fi
    # e2e.py takes one --impl at a time (or "all"); it also brings up Postgres
    # and loads the snapshot, so the first arm pays for the dataset.
    if [[ "$(echo $YTNAV_ARMS | tr ' ' '\n' | sort | tr '\n' ' ')" == "byllm langgraph openai_sdk " ]]; then
        run_step ytnav all "$RUN_DIR/ytnav/all" \
            "$YTNAV_PY" "$ROOT/YTNavigator/eval/e2e.py" --impl all \
            --langgraph-python "$YTNAV_PY" --questions "$questions" ${JUDGE:+--judge}
    else
        for arm in $YTNAV_ARMS; do
            run_step ytnav "$arm" "$RUN_DIR/ytnav/$arm" \
                "$YTNAV_PY" "$ROOT/YTNavigator/eval/e2e.py" --impl "$arm" \
                --langgraph-python "$YTNAV_PY" --questions "$questions" ${JUDGE:+--judge}
        done
    fi
    [[ -n "$DRY_RUN" ]] && return 0
    archive "$RUN_DIR/ytnav/results" "$ROOT/YTNavigator/eval/out"
}

# The langgraph CodeAgent arm pins langchain>=1.2 / langgraph>=1.0, which no
# shared environment carries (YT-Navigator pins the 0.3 line), so it keeps its
# own venv the way the two CrewAI arms do. $CODE_PY_<arm> overrides per arm.
codeagent_python() {
    local arm="$1" var venv
    var="CODE_PY_${arm}"
    [[ -n "${!var:-}" ]] && { echo "${!var}"; return; }
    venv="$ROOT/CodeAgent/langgraph/.venv/bin/python"
    [[ "$arm" == langgraph && -x "$venv" ]] && { echo "$venv"; return; }
    # NOOA imports the `nooa` package, which only the nooa conda env carries.
    venv="$HOME/miniconda3/envs/nooa/bin/python"
    [[ "$arm" == nooa && -x "$venv" ]] && { echo "$venv"; return; }
    echo "$PY"
}

sweep_codeagent() {
    local arm run_dirs=() outdir="$RUN_DIR/codeagent/swebench" apy
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    for arm in $CODEAGENT_ARMS; do
        apy="$(codeagent_python "$arm")"
        run_step codeagent "$arm" "$RUN_DIR/codeagent/$arm" \
            "$PY" "$ROOT/CodeAgent/swebench_bridge/run_agent.py" --framework "$arm" \
            --instance-ids $CODE_INSTANCES --run-id "$RUN_ID-$arm" --output-dir "$outdir" \
            --runtime "$CODE_RUNTIME" --workers "$CODE_WORKERS" --max-steps "$CODE_MAX_STEPS" \
            --model "$CODEAGENT_MODEL" --jac "$JAC_BIN/jac" --python "$apy" ${CODE_FORCE:+--force}
        run_dirs+=("$outdir/$RUN_ID-$arm")
        # run_agent.py exits 0 even when every instance died in import; the only
        # honest success signal is whether any patch came out. Read it from the
        # predictions file, last row per instance, not from driver.log: that log
        # is appended across re-runs, so an earlier attempt's summary would be
        # matched instead of this one's.
        if [[ -z "$DRY_RUN" ]] && [[ "$(python3 - "$outdir/$RUN_ID-$arm/predictions.jsonl" <<'EOF'
import json, sys
rows = {}
try:
    for line in open(sys.argv[1]):
        if line.strip():
            r = json.loads(line); rows[r["instance_id"]] = r
except FileNotFoundError:
    pass
print(sum(1 for r in rows.values() if (r.get("model_patch") or "").strip()))
EOF
)" == 0 ]]; then
            log "codeagent/$arm produced 0 patches -- treating as FAILED"
            sed -i "s|^codeagent\t$arm\tok|codeagent\t$arm\tfailed:no-patches|" "$RUN_DIR/summary.tsv"
        fi
    done
    [[ -n "$DRY_RUN" ]] && return 0
    local d
    for d in "${run_dirs[@]}"; do
        [[ -f "$d/predictions.jsonl" ]] || { log "codeagent: no predictions in $d, not graded"; continue; }
        # grade.py resumes from eval_results.jsonl, so after a --force re-run it
        # would hand back the previous attempt's verdicts without running a test.
        [[ -n "${CODE_FORCE:-}" ]] && rm -f "$d/eval_results.jsonl"
        log "codeagent grading $(basename "$d")"
        "$PY" "$ROOT/CodeAgent/swebench_bridge/grade.py" --predictions "$d/predictions.jsonl" \
            --runtime "$CODE_RUNTIME" >> "$RUN_DIR/codeagent/grade.log" 2>&1 \
            || log "codeagent grading failed for $(basename "$d")"
    done
    "$PY" "$ROOT/CodeAgent/swebench_bridge/report.py" "${run_dirs[@]}" \
        > "$RUN_DIR/codeagent/report.txt" 2>&1 || log "codeagent report failed"
}

# ---------------------------------------------------------------------- main
mkdir -p "$RUN_DIR"
: > "$RUN_DIR/summary.tsv"
preflight | tee "$RUN_DIR/preflight.txt" || {
    echo "preflight failed; fix the items above or set SWEEP_SKIP_PREFLIGHT=1" >&2
    [[ -z "${SWEEP_SKIP_PREFLIGHT:-}" ]] && exit 3
    true
}
[[ -n "$CHECK_ONLY" ]] && exit 0

log "sweep start: model=$SWEEP_MODEL apps=${APPS[*]}"
for app in "${APPS[@]}"; do
    case "$app" in
        factcheck) sweep_factcheck ;;
        email)     sweep_email ;;
        meeting)   sweep_meeting ;;
        ytnav)     sweep_ytnav ;;
        codeagent) sweep_codeagent ;;
        *) echo "unknown agent '$app' (factcheck email meeting ytnav codeagent)" >&2; exit 2 ;;
    esac
done

log "sweep done -> $RUN_DIR"
echo
printf '%-11s %-18s %-12s %8s\n' app arm status secs
awk -F'\t' '{printf "%-11s %-18s %-12s %8s\n", $1, $2, $3, $4}' "$RUN_DIR/summary.tsv"
