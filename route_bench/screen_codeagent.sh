#!/usr/bin/env bash
# Screen SWE-bench Lite instances with the cheap model only, in batches of 3 so images are graded and retired
# before the next pull (disk is tight). Usage: screen_codeagent.sh <ids-file>
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ids=($(grep -v '^\s*$' "$1"))
export RESULTS_NAME=codeagent_screen
for ((i=0; i<${#ids[@]}; i+=3)); do
  batch=("${ids[@]:i:3}")
  echo "[$(date '+%F %T')] screen batch: ${batch[*]}"
  EVAL="${batch[*]}" "$ROOT/route_bench/run_arm.sh" codeagent eval cheap 2>&1 | tail -3
  "$ROOT/route_bench/run_arm.sh" codeagent score 2>&1 | tail -2
  grep -E "^\[.*\] (RESOLVED|UNRESOLVED|ERROR|EMPTY)" "$ROOT/route_bench/results/codeagent_screen/eval/cheap/grade.log" | sort -u
  df -h / | tail -1 | awk '{print "disk free", $4}'
done
echo "[$(date '+%F %T')] SCREEN_DONE"
