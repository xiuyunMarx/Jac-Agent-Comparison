#!/usr/bin/env python
"""Profile the vLLM-backed Jac-Rag-GPT: how much of e2e turn time is prefill.

Drives Jac-Rag-GPT-vllm over REST (reusing drivers.JacServeDriver) against a
running vLLM server on :8001, and attributes per-turn server-side time by
scraping /metrics before and after each turn (driving is sequential, so the
histogram deltas belong to that turn alone).

Usage:
    python profile_vllm.py --limit 30            # stratified over categories
    python profile_vllm.py --categories rag_qa,small_talk --limit 4
Prereq: Jac-Rag-GPT-vllm/serve_vllm.sh is up (vLLM on :8001).
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402
import drivers  # noqa: E402

VLLM_METRICS_URL = "http://127.0.0.1:8001/metrics"
PROJECT_DIR = common.CODER_DIR / "Jac-Rag-GPT-vllm"
OUT_DIR = common.RESULTS_DIR / "vllm_profile"

# Histograms read via their _sum/_count series; counters read directly. Names
# matched by substring so minor renames across vLLM versions don't break us.
HIST_KEYS = {
    "prefill": "request_prefill_time_seconds",
    "decode": "request_decode_time_seconds",
    "queue": "request_queue_time_seconds",
    "ttft": "time_to_first_token_seconds",
    "e2e_llm": "e2e_request_latency_seconds",
    "kv_computed": "request_prefill_kv_computed_tokens",
}
COUNTER_KEYS = {
    "prompt_tokens": "prompt_tokens_total",
    "gen_tokens": "generation_tokens_total",
    "cache_queries": "prefix_cache_queries_total",
    "cache_hits": "prefix_cache_hits_total",
}

METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+|NaN)$")


def scrape() -> dict[str, float]:
    """Sum every series per metric name (labels collapsed)."""
    text = requests.get(VLLM_METRICS_URL, timeout=10).text
    acc: dict[str, float] = defaultdict(float)
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = METRIC_LINE.match(line)
        if m and m.group(2) != "NaN":
            acc[m.group(1)] += float(m.group(2))
    return dict(acc)


def pick(snap: dict[str, float], fragment: str, suffix: str = "") -> float:
    want = fragment + suffix
    for name, val in snap.items():
        if name.endswith(want) or want in name:
            return val
    return 0.0


def turn_delta(before: dict, after: dict) -> dict:
    row = {}
    for key, frag in HIST_KEYS.items():
        row[f"{key}_s" if "tokens" not in frag else key] = (
            pick(after, frag, "_sum") - pick(before, frag, "_sum"))
    row["n_llm_calls"] = int(pick(after, HIST_KEYS["prefill"], "_count")
                             - pick(before, HIST_KEYS["prefill"], "_count"))
    for key, frag in COUNTER_KEYS.items():
        row[key] = pick(after, frag) - pick(before, frag)
    return row


def sample_items(categories: list[str], limit: int) -> list[dict]:
    data = common.read_jsonl(common.DATASET_PATH)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for it in data:
        if it["category"] in categories:
            by_cat[it["category"]].append(it)
    cats = [c for c in categories if by_cat.get(c)]
    picked, i = [], 0
    while len(picked) < limit and any(by_cat[c] for c in cats):
        c = cats[i % len(cats)]
        if by_cat[c]:
            picked.append(by_cat[c].pop(0))
        i += 1
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--categories", default=",".join(common.CATEGORIES))
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--fresh", action="store_true",
                    help="discard turns.jsonl from previous runs")
    args = ap.parse_args()

    try:
        scrape()
    except requests.RequestException:
        raise SystemExit("vLLM server not reachable on :8001 — run "
                         "Jac-Rag-GPT-vllm/serve_vllm.sh first")

    items = sample_items([c.strip() for c in args.categories.split(",")], args.limit)
    print(f"{len(items)} items sampled")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    turns_path = OUT_DIR / "turns.jsonl"
    if args.fresh:
        turns_path.unlink(missing_ok=True)

    driver = drivers.JacServeDriver("jac-vllm", PROJECT_DIR, args.port, OUT_DIR)
    driver.start()
    try:
        for idx, item in enumerate(items):
            session = f"prof_{idx}_{int(time.time())}"
            for tno, turn in enumerate(item["turns"]):
                before = scrape()
                t0 = time.perf_counter()
                err, agent, resp_len = "", "", 0
                try:
                    result = driver.interact(turn["message"], session)
                    agent = result["agent"]
                    resp_len = len(result["response"])
                except Exception as e:  # keep profiling through bad turns
                    err = str(e)[:200]
                client_e2e = time.perf_counter() - t0
                row = {
                    "item": idx, "turn": tno, "category": item["category"],
                    "gold_agent": turn.get("gold_agent", ""), "agent": agent,
                    "client_e2e_s": round(client_e2e, 4),
                    "resp_len": resp_len, "error": err,
                    **{k: round(v, 4) if isinstance(v, float) else v
                       for k, v in turn_delta(before, scrape()).items()},
                }
                common.append_jsonl(turns_path, row)
                print(f"[{idx:>3}.{tno}] {item['category']:<11} agent={agent:<13} "
                      f"e2e={client_e2e:6.2f}s calls={row['n_llm_calls']} "
                      f"prefill={row['prefill_s']:.3f}s decode={row['decode_s']:.3f}s"
                      + (f"  ERROR {err}" if err else ""))
    finally:
        driver.stop()

    write_report(common.read_jsonl(turns_path))


def write_report(rows: list[dict]) -> None:
    ok = [r for r in rows if not r["error"]]
    if not ok:
        print("no successful turns; no report")
        return

    def agg(sub: list[dict]) -> dict:
        s = lambda k: sum(r.get(k, 0) for r in sub)
        llm = s("prefill_s") + s("decode_s")
        return {
            "turns": len(sub),
            "calls/turn": s("n_llm_calls") / len(sub),
            "prefill_s": s("prefill_s"), "decode_s": s("decode_s"),
            "queue_s": s("queue_s"), "client_e2e_s": s("client_e2e_s"),
            "prefill/llm": s("prefill_s") / llm if llm else 0.0,
            "prefill/e2e": s("prefill_s") / s("client_e2e_s") if s("client_e2e_s") else 0.0,
            "llm/e2e": llm / s("client_e2e_s") if s("client_e2e_s") else 0.0,
            "prompt_tok": s("prompt_tokens"), "kv_computed_tok": s("kv_computed"),
            "gen_tok": s("gen_tokens"),
            "cache_hit_rate": (s("cache_hits") / s("cache_queries")
                               if s("cache_queries") else 0.0),
        }

    by_cat = defaultdict(list)
    for r in ok:
        by_cat[r["category"]].append(r)

    lines = ["# Jac-Rag-GPT on vLLM (Qwen2.5-3B) — prefill share profile", ""]
    lines.append(f"{len(ok)} turns ok, {len(rows) - len(ok)} errored. "
                 "Server-side times from vLLM /metrics deltas per turn; "
                 "client e2e includes routing, RAG search, reranking, framework.")
    lines.append("")
    hdr = ("| scope | turns | calls/turn | prefill s | decode s | client e2e s | "
           "prefill/LLM | prefill/e2e | LLM/e2e | KV computed/prompt tok | prefix-cache hit |")
    lines += [hdr, "|" + "---|" * 11]
    for name, sub in [("**overall**", ok)] + sorted(by_cat.items()):
        a = agg(sub)
        kvr = a["kv_computed_tok"] / a["prompt_tok"] if a["prompt_tok"] else 0.0
        lines.append(
            f"| {name} | {a['turns']} | {a['calls/turn']:.1f} | {a['prefill_s']:.2f} "
            f"| {a['decode_s']:.2f} | {a['client_e2e_s']:.2f} | {a['prefill/llm']:.1%} "
            f"| {a['prefill/e2e']:.1%} | {a['llm/e2e']:.1%} | {kvr:.1%} "
            f"| {a['cache_hit_rate']:.1%} |")
    lines += ["", "- `prefill/LLM`: prefill time over prefill+decode (pure LLM view).",
              "- `prefill/e2e`: prefill time over client wall time — the share "
              "proactive prefill can attack end-to-end.",
              "- `KV computed/prompt tok`: <100% means vLLM's reactive prefix cache "
              "already skipped some prefill; proactive prefill targets the rest.", ""]
    report = "\n".join(lines)
    (OUT_DIR / "report.md").write_text(report)
    print("\n" + report)
    print(f"written: {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
