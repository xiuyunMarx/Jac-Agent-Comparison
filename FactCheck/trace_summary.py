"""Summarise FC_TRACE logs from the OpenaiSDK / LangGraph arms.

    python trace_summary.py trace.jsonl [more.jsonl ...]

Per file: calls, retries, prompt/completion tokens by stage, prompt-token
growth per round, and the largest single reply -- the numbers that show a
runaway (retry storms, oversized outputs, unexpected prompt growth).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def summarise(path: str) -> None:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows:
        print(f"{path}: empty")
        return
    by_stage: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_stage[r["stage"]].append(r)

    def tok(r: dict, key: str) -> int:
        return int((r.get("usage") or {}).get(key) or 0)

    total_p = sum(tok(r, "prompt_tokens") for r in rows)
    total_c = sum(tok(r, "completion_tokens") for r in rows)
    retries = sum(1 for r in rows if r["stage"].endswith("/retry"))
    rounds = max(r["round"] for r in rows)
    print(f"== {path}")
    print(f"calls {len(rows)}  retries {retries}  rounds {rounds}  prompt_tokens {total_p}  completion_tokens {total_c}")
    print(f"{'stage':16} {'n':>3} {'prompt_tok':>10} {'compl_tok':>9} {'avg_prompt_chars':>16} {'max_reply_chars':>15}")
    for stage in sorted(by_stage):
        rs = by_stage[stage]
        print(
            f"{stage:16} {len(rs):>3} {sum(tok(r, 'prompt_tokens') for r in rs):>10} "
            f"{sum(tok(r, 'completion_tokens') for r in rs):>9} "
            f"{sum(r['prompt_chars'] for r in rs) // len(rs):>16} {max(r['reply_chars'] for r in rs):>15}"
        )
    print("prompt tokens of the verify call per round:",
          [tok(r, "prompt_tokens") for r in rows if r["stage"] == "verify"])
    biggest = max(rows, key=lambda r: r["reply_chars"])
    print(f"largest reply: {biggest['reply_chars']} chars in {biggest['stage']} round {biggest['round']}")
    if any(r.get("usage") is None for r in rows):
        print("note: some calls carried no usage block")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarise(p)
