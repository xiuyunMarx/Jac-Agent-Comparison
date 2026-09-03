#!/usr/bin/env python3
"""Score prefix-cache reuse from byLLM request logs, the way an SGLang-style
radix cache would see it.

Input: one or more BYLLM_PROMPT_LOG files (one JSON object per completion call:
the request as sent -- messages, tools, response_format -- plus the usage record).

Model: the server keeps every prompt it has served in a radix tree of tokens.
A new request reuses the longest token prefix it shares with any earlier prompt
(no cache_control markers, no minimum size); only the remaining tail is
prefilled. Reported per run:

  hit rate   = cached tokens / prompt tokens     (what SGLang reports as cached_tokens)
  prefill    = 1 - hit rate                      (tokens that still had to be computed)
  invariant  = the PR's own metric: tokens of the hoisted prefix / prompt tokens

The prompt is rendered in the order a chat template sees it: tools, then the
system messages, then the conversation. Each log file is one session; with
--shared all sessions of a run share one cache in timestamp order (one server
serving all workers), otherwise each session starts cold. --ttl SECONDS evicts
entries not touched for that long (default: never, like an uncapped radix cache).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

_ENC = None


def encode(text: str, model: str) -> list[int]:
    """Token ids via litellm's tokenizer when importable, else chars/4 pseudo-tokens."""
    global _ENC
    if _ENC is None:
        try:
            import litellm  # noqa
            _ENC = lambda t, m: litellm.encode(model=m or "gpt-4o", text=t)  # noqa: E731
        except Exception:
            _ENC = lambda t, m: [ord(c) for c in t[::4]]  # noqa: E731
    if not text:
        return []
    try:
        return list(_ENC(text, model))
    except Exception:
        return [ord(c) for c in text[::4]]


def render_prompt(req: dict) -> str:
    """Flatten the request in template order: tools, system messages, conversation."""
    parts: list[str] = []
    for tool in req.get("tools") or []:
        parts.append("TOOL:" + json.dumps({k: v for k, v in tool.items() if k != "cache_control"}, sort_keys=True))
    msgs = [m if isinstance(m, dict) else {"role": "assistant", "content": str(m)}
            for m in (req.get("messages") or [])]

    def text_of(m: dict) -> str:
        c = m.get("content")
        if isinstance(c, str) or c is None:
            body = c or ""
        else:
            body = "\n".join(json.dumps({k: v for k, v in b.items() if k != "cache_control"}, sort_keys=True)
                             if isinstance(b, dict) else str(b) for b in c)
        if m.get("tool_calls"):
            body += json.dumps(m["tool_calls"], sort_keys=True, default=str)
        return f"{m.get('role')}:{body}"

    parts.extend(text_of(m) for m in msgs if m.get("role") == "system")
    parts.extend(text_of(m) for m in msgs if m.get("role") != "system")
    return "\n".join(parts)


def lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def score(records: list[dict], ttl: float | None) -> tuple[dict, list[dict]]:
    """Score calls in order against one cache. Returns (totals, per-call rows)."""
    cache: list[tuple[list[int], float]] = []      # (token ids, last touch)
    rows: list[dict] = []
    tot: collections.Counter = collections.Counter()
    for r in records:
        model = r.get("model", "")
        ts = float(r.get("ts", 0.0))
        toks = encode(render_prompt(r), model)
        if ttl is not None:
            cache = [(t, at) for t, at in cache if ts - at <= ttl]
        best, best_i = 0, -1
        for i, (prev, _) in enumerate(cache):
            n = lcp(prev, toks)
            if n > best:
                best, best_i = n, i
        if best_i >= 0:
            cache[best_i] = (cache[best_i][0], ts)   # a hit keeps the entry warm
        cache.append((toks, ts))
        inv = int(r.get("invariant_tokens") or 0)
        row = {"scope": r.get("scope", ""), "hoisted": bool(r.get("hoisted")),
               "prompt_tokens_provider": int(r.get("prompt_tokens") or 0),
               "prompt_tokens": len(toks), "cached": best, "prefill": len(toks) - best,
               "invariant_tokens": inv, "n_messages": len(r.get("messages") or [])}
        rows.append(row)
        for k in ("prompt_tokens_provider", "prompt_tokens", "cached", "prefill", "invariant_tokens"):
            tot[k] += row[k]
        tot["calls"] += 1
    return dict(tot), rows


def summarize(name: str, tot: dict, sessions: int) -> dict:
    p = tot.get("prompt_tokens", 0) or 1
    return {"run": name, "sessions": sessions, "calls": tot.get("calls", 0),
            "prompt_tokens": tot.get("prompt_tokens", 0),
            "prompt_tokens_provider": tot.get("prompt_tokens_provider", 0),
            "cached_tokens": tot.get("cached", 0), "prefill_tokens": tot.get("prefill", 0),
            "hit_rate": round(tot.get("cached", 0) / p, 4),
            "invariant_frac": round(tot.get("invariant_tokens", 0) / p, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="prompt-log files as NAME=path; repeat a NAME to pool sessions")
    ap.add_argument("--shared", action="store_true", help="all sessions of a run share one cache (timestamp order)")
    ap.add_argument("--ttl", type=float, default=None, help="evict entries idle for this many seconds (default: never)")
    ap.add_argument("--per-call", action="store_true")
    ap.add_argument("--per-scope", action="store_true", help="break the hit rate down by ability")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    runs: dict[str, list[list[dict]]] = collections.OrderedDict()
    for spec in args.logs:
        name, _, path = spec.rpartition("=")
        name = name or os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]
        recs.sort(key=lambda r: float(r.get("ts", 0)))
        runs.setdefault(name, []).append(recs)

    summaries, per_scope, per_call = [], {}, {}
    for name, sessions in runs.items():
        tot: collections.Counter = collections.Counter()
        rows: list[dict] = []
        groups = [sorted((r for s in sessions for r in s), key=lambda r: float(r.get("ts", 0)))] if args.shared else sessions
        for recs in groups:
            t, rws = score(recs, args.ttl)
            tot.update(t)
            rows.extend(rws)
        summaries.append(summarize(name, dict(tot), len(sessions)))
        per_call[name] = rows
        sc: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for row in rows:
            c = sc[row["scope"].split(".")[-1]]
            c["calls"] += 1
            for k in ("prompt_tokens", "cached", "invariant_tokens"):
                c[k] += row[k]
        per_scope[name] = sc

    if args.json:
        print(json.dumps(summaries, indent=2))
        return 0
    hdr = f"{'run':24} {'sess':>4} {'calls':>5} {'prompt tok':>10} {'cached':>9} {'prefill':>9} {'hit rate':>8} {'invariant':>9} {'provider tok':>12}"
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        print(f"{s['run']:24} {s['sessions']:>4} {s['calls']:>5} {s['prompt_tokens']:>10} {s['cached_tokens']:>9} "
              f"{s['prefill_tokens']:>9} {s['hit_rate']:>8.1%} {s['invariant_frac']:>9.1%} {s['prompt_tokens_provider']:>12}")
    if args.per_scope:
        print("\nper ability:")
        for name, sc in per_scope.items():
            for scope, c in sorted(sc.items()):
                p = c["prompt_tokens"] or 1
                print(f"  {name:12} {scope:26} calls={c['calls']:4} tokens={c['prompt_tokens']:8} "
                      f"hit={c['cached'] / p:6.1%} invariant={c['invariant_tokens'] / p:6.1%}")
    if args.per_call:
        for name, rows in per_call.items():
            print(f"\n== {name}")
            for i, row in enumerate(rows):
                print(f"  c{i:03d} {row['scope'][-36:]:36} tok={row['prompt_tokens']:>6} cached={row['cached']:>6} "
                      f"prefill={row['prefill']:>6} inv={row['invariant_tokens']:>5} msgs={row['n_messages']} hoist={int(row['hoisted'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
