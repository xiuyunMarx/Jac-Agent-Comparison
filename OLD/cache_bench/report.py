#!/usr/bin/env python3
"""Merge cache_bench/results/<arm>/summary.json files into one markdown table."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results")
arms = {}
for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    f = arm_dir / "summary.json"
    if f.is_file():
        arms[arm_dir.name] = {s["run"]: s for s in json.loads(f.read_text())}

print("| arm | mode | sessions | calls | prompt tokens | cached | prefill | hit rate | invariant |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for arm, d in arms.items():
    for run in ("baseline", "hoisted"):
        s = d.get(run)
        if s:
            print(f"| {arm} | {run} | {s['sessions']} | {s['calls']} | {s['prompt_tokens']:,} | {s['cached_tokens']:,} | "
                  f"{s['prefill_tokens']:,} | {s['hit_rate']:.1%} | {s['invariant_frac']:.1%} |")
print()
print("| arm | hit rate baseline -> hoisted | change | prefill tokens baseline -> hoisted |")
print("|---|---|---:|---|")
for arm, d in arms.items():
    if "baseline" in d and "hoisted" in d:
        b, h = d["baseline"], d["hoisted"]
        print(f"| {arm} | {b['hit_rate']:.1%} -> {h['hit_rate']:.1%} | {h['hit_rate'] - b['hit_rate']:+.1%} | "
              f"{b['prefill_tokens']:,} -> {h['prefill_tokens']:,} |")
