#!/usr/bin/env python3
"""Hit rate on 'cold' requests only: the first request of an ability invocation (no
assistant turn in the prompt yet). Later turns of an agent loop are dominated by
the conversation's own prefix, which hoisting does not touch; the cold request is
where a hoisted invariant block can be shared across invocations and abilities."""
import glob, json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results")
def load(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    recs.sort(key=lambda r: float(r.get("ts", 0))); return recs
def is_cold(r):
    return not any((m.get("role") if isinstance(m, dict) else "") == "assistant" for m in r.get("messages") or [])

print(f"{'arm':10} {'mode':9} {'cold calls':>10} {'cold tok':>9} {'cold hit':>9} {'all hit':>8}")
for arm in sorted(p.name for p in root.iterdir() if p.is_dir()):
    for mode in ("baseline", "hoisted"):
        d = root / arm / mode
        files = sorted(glob.glob(str(d / "swebench" / f"cache-{mode}-jac" / "logs" / "*" / "byllm_prompts.jsonl"))) or [d / "prompts.jsonl"]
        files = [f for f in files if Path(f).is_file()]
        if not files: continue
        recs = sorted((r for f in files for r in load(f)), key=lambda r: float(r.get("ts", 0)))  # one shared server
        tot, rows = S.score(recs, None)
        cold = [row for r, row in zip(recs, rows) if is_cold(r)]
        ct = sum(x["prompt_tokens"] for x in cold); cc = sum(x["cached"] for x in cold)
        print(f"{arm:10} {mode:9} {len(cold):>10} {ct:>9} {cc/ct if ct else 0:>9.1%} {tot['cached']/tot['prompt_tokens']:>8.1%}")
