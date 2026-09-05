#!/usr/bin/env bash
# Detached full evaluation: serial 3-arm pass, one retry pass for failed runs, then scoring.
# Progress: eval/out/run.log   Report: eval/out/report.md   Finished marker: eval/out/DONE
cd "$(dirname "$0")/.."
set -a; . ../.env; set +a
PY=${FC_PY:-$HOME/miniconda3/envs/jaseci/bin/python}
OUT="$PWD/eval/out"; mkdir -p "$OUT"; rm -f "$OUT/DONE"
echo "[$(date '+%F %T')] pass 1" >> "$OUT/run.log"
$PY eval/run.py --serial --out "$OUT" >> "$OUT/run.log" 2>&1
echo "[$(date '+%F %T')] pass 2 (retry failed)" >> "$OUT/run.log"
$PY eval/run.py --serial --out "$OUT" >> "$OUT/run.log" 2>&1
echo "[$(date '+%F %T')] scoring" >> "$OUT/run.log"
$PY eval/score.py --out "$OUT" > "$OUT/score.log" 2>&1
echo "[$(date '+%F %T')] done" >> "$OUT/run.log"
touch "$OUT/DONE"
