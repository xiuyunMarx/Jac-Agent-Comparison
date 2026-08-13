#!/usr/bin/env bash
# Repair verification for django__django-12983.
#
#   ./verify.sh                  apply gold.patch, run the graded tests, print the verdict
#   ./verify.sh my_fix.diff      same, with your own patch
#   ./verify.sh --none           baseline: no fix. In container mode the harness reports an
#                                empty patch as NOT RESOLVED without running the suite; pair it
#                                with --local to actually watch FAIL_TO_PASS fail.
#   ./verify.sh --local [patch]  run against ./repo in the current shell instead of a container
#
# Default runs the official SWE-bench eval script inside
#   swebench/sweb.eval.x86_64.django_1776_django-12983:latest
# via swebench_bridge/grade_local.py + udocker. The image is pulled on first use (~1-2 GB)
# and carries the exact interpreter and dependencies this instance needs.
#
# --local runs ./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 utils_tests.test_text
# against ./repo directly. That only works if the active environment already matches
# django/django 3.2 (Python version included); most cases here need the container.
set -euo pipefail
CASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE="$CASE/../../../swebench_bridge"

MODE=container
if [ "${1:-}" = "--local" ]; then MODE=local; shift; fi
PATCH="${1:-$CASE/gold.patch}"
if [ "$PATCH" = "--none" ]; then PATCH=""; else
  PATCH="$(cd "$(dirname "$PATCH")" && pwd)/$(basename "$PATCH")"
fi

if [ "$MODE" = container ]; then
  PRED="$(mktemp /tmp/django__django-12983.XXXX.jsonl)"
  python3 - "$PATCH" > "$PRED" <<'PY'
import json, sys
p = sys.argv[1] if len(sys.argv) > 1 else ""
print(json.dumps({"instance_id": "django__django-12983",
                  "model_patch": open(p).read() if p else "",
                  "model_name_or_path": "verify"}))
PY
  python3 "$BRIDGE/grade_local.py" --predictions "$PRED" --instance-ids django__django-12983 \
          --run-id "verify-django__django-12983" --workers 1
  echo
  echo "report: $(dirname "$PRED")/verify.verify-django__django-12983.json"
  exit 0
fi

cd "$CASE/repo"
git checkout -q -- . && git clean -qfd            # back to the buggy base commit
git apply "$CASE/test_patch.diff"                  # the graded tests
[ -n "$PATCH" ] && git apply "$PATCH"

echo "--- ./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 utils_tests.test_text"
set +e
PYTHONPATH="$CASE/repo${PYTHONPATH:+:$PYTHONPATH}" \
  ./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 utils_tests.test_text > "$CASE/test_output.txt" 2>&1
set -e
python3 "$CASE/../../grade.py" "$CASE" "$CASE/test_output.txt"
