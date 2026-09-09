#!/usr/bin/env python3
"""Summarize the routing bench: task quality and cost per mode, and where the router sent each site.

Reads results/eval/<mode>/{mock_output,usage.jsonl} and results/router/summary.json,
scores the runs with Email-Auto-response/eval/score.py (deterministic metrics only)
and prices every LLM call from pricing.json. Writes results/REPORT.md.
"""
import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRICES = json.loads((HERE / "pricing.json").read_text())


def bare(m):
    return m.split("/", 1)[1] if "/" in m else m


def cost(model, pt, ct):
    p = PRICES.get(bare(model))
    return None if not p else (pt * p[0] + ct * p[2]) / 1e6


def usage_stats(mode_dir):
    """Calls/tokens/cost from the harness token meter in each results file (priced from pricing.json),
    and the per-site model split from the runtime usage log if present, else from the router log."""
    tot = collections.Counter()
    for f in sorted((mode_dir / "mock_output").glob("results_*.json")):
        u = json.load(open(f)).get("usage") or {}
        for model, rec in (u.get("by_model") or {}).items():
            pt, ct = int(rec.get("prompt_tokens", 0)), int(rec.get("completion_tokens", 0))
            tot["calls"] += int(rec.get("calls", 0)); tot["prompt_tokens"] += pt; tot["completion_tokens"] += ct
            tot["cost"] += cost(model, pt, ct) or 0.0
    by_site = collections.defaultdict(lambda: collections.Counter())
    src = mode_dir / "usage.jsonl"
    if src.exists():
        for line in open(src):
            r = json.loads(line)
            by_site[r["scope"].rsplit(".", 1)[-1]][bare(r["model"])] += 1
            tot["retries"] += int(r.get("route_reason") == "escalate")
    elif (mode_dir / "router.jsonl").exists():
        for line in open(mode_dir / "router.jsonl"):
            r = json.loads(line)
            by_site[(r.get("site") or "").rsplit(".", 1)[-1]][bare(r["model"])] += 1
    return by_site, tot


def score_run(mode_dir):
    """Deterministic email metrics via the arm's own scorer (no LLM judge)."""
    sys.path.insert(0, str(ROOT / "Email-Auto-response" / "eval"))
    import score as email_score
    runs = []
    for results_path in sorted((mode_dir / "mock_output").glob("results_*.json")):
        results = email_score.load_json(results_path)
        dataset = email_score.load_json(email_score.resolve_dataset(results, results_path))
        runs.append(email_score.score_run(results, dataset))
    (mode_dir / "scores.json").write_text(json.dumps(runs, indent=2))
    return runs, ""


def judge_stats(mode_dir):
    """Pooled LLM-judge numbers from the scores_<label>_<batch>.json files the judge step copies here."""
    files = sorted(mode_dir.glob("scores_*_batch_*.json"))
    n = kp_c = kp_t = 0; tone = fact = overall = 0.0
    for f in files:
        j = json.loads(f.read_text()).get("judge") or {}
        for d in j.get("per_draft", []):
            if "error" in d:
                continue
            n += 1; tone += d["tone_match"]; fact += d["factuality"]; overall += d["overall"]
            kps = d.get("key_points", [])
            kp_t += len(kps); kp_c += sum(bool(k.get("covered")) for k in kps)
    if not n:
        return None
    return {"drafts": n, "kp": kp_c / kp_t if kp_t else None, "tone": tone / n, "fact": fact / n, "overall": overall / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--eval", default=None)
    ap.add_argument("--router", default=None)
    args = ap.parse_args()
    res = Path(args.results)
    eval_dir = Path(args.eval) if args.eval else res / "eval"
    router_dir = Path(args.router) if args.router else res / "router"
    lines = ["# Learnt model routing on the Email byLLM arm", ""]
    rs = router_dir / "summary.json"
    if rs.exists():
        s = json.loads(rs.read_text())
        lines += [f"Router: LLMRouter KNNRouter (k={s.get('k', 5)}, Longformer embeddings), trained on "
                  f"{s['train_requests']} requests replayed on {', '.join(bare(c) for c in s['candidates'])}; "
                  f"held-out oracle accuracy {s['router_oracle_accuracy']:.0%} on {s['test_requests']} requests "
                  f"(lambda={s['lambda']}).", ""]
    rows = []
    for mode in ("strong", "cheap", "routed"):
        d = eval_dir / mode
        if not (d / "mock_output").exists():
            continue
        by_site, tot = usage_stats(d)
        scores, _ = score_run(d)
        rows.append((mode, by_site, tot, scores))
    lines += ["| mode | runs | expected replies found (tp/fp/fn) | filtering P / R / F1 | drafts done / correct recipient | LLM calls | prompt tok | completion tok | cost USD |",
              "|---|---:|---|---|---|---:|---:|---:|---:|"]
    for mode, by_site, tot, scores in rows:
        runs = scores if isinstance(scores, list) else scores.get("runs", scores)
        f = agg(runs)
        lines.append(f"| {mode} | {f['n']} | {f['tp']}/{f['fp']}/{f['fn']} | {f['p']:.2f} / {f['r']:.2f} / {f['f1']:.2f} | {f['done']:.2f} / {f['recip']:.2f} | "
                     f"{tot['calls']} | {tot['prompt_tokens']:,} | {tot['completion_tokens']:,} | {tot['cost']:.4f} |")
    judged = [(mode, judge_stats(eval_dir / mode)) for mode, *_ in rows]
    if any(j for _, j in judged):
        lines += ["", "LLM judge (glm-5.2 via ollama.com; 1-5 scales, key-point coverage as a fraction):", "",
                  "| mode | drafts judged | key points covered | tone | factuality | overall |", "|---|---:|---:|---:|---:|---:|"]
        for mode, j in judged:
            if j:
                kp = f"{j['kp']:.2f}" if j["kp"] is not None else "-"
                lines.append(f"| {mode} | {j['drafts']} | {kp} | {j['tone']:.2f} | {j['fact']:.2f} | {j['overall']:.2f} |")
    lines += ["", "## Where each call site went (calls per model)", ""]
    for mode, by_site, tot, _ in rows:
        lines.append(f"**{mode}**" + (f" (typed-output retries escalated: {tot['retries']})" if tot['retries'] else ""))
        for site, ms in sorted(by_site.items()):
            lines.append(f"- {site}: " + ", ".join(f"{m} x{n}" for m, n in ms.most_common()))
        lines.append("")
    (eval_dir / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


def agg(runs):
    """Mean of the deterministic metrics over runs, tolerant of the scorer's shapes."""
    n = 0; p = r = f1 = done = recip = 0.0; n_recip = 0; tp = fp = fn = 0
    items = runs.values() if isinstance(runs, dict) else runs
    for run in items:
        if not isinstance(run, dict) or "filtering" not in run:
            continue
        n += 1
        flt, dr = run["filtering"], run.get("drafts", {})
        p += flt.get("precision", 0.0); r += flt.get("recall", 0.0); f1 += flt.get("f1", 0.0)
        done += dr.get("completion_rate", 0.0) or 0.0
        if dr.get("correct_recipient_rate") is not None:
            recip += dr["correct_recipient_rate"]; n_recip += 1
        tp += flt.get("true_positives", 0); fp += flt.get("false_positives", 0); fn += flt.get("false_negatives", 0)
    n = max(n, 1)
    # pooled precision/recall over all batches (per-batch F1 is 1.0 by convention when nothing is expected)
    pp = tp / (tp + fp) if tp + fp else 1.0
    pr = tp / (tp + fn) if tp + fn else 1.0
    pf = 2 * pp * pr / (pp + pr) if pp + pr else 0.0
    return {"n": n, "p": pp, "r": pr, "f1": pf, "done": done / n, "recip": recip / max(n_recip, 1),
            "tp": tp, "fp": fp, "fn": fn}


if __name__ == "__main__":
    main()
