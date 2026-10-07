#!/usr/bin/env python3
"""Cost + quality per mode for the non-email arms (results/<arm>/eval/<mode>), as markdown.

Cost is priced from the runtime usage logs (served model per call, pricing.json).
Quality comes from each arm's own scorer output written by run_arm.sh score.
"""
import argparse
import collections
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRICES = json.loads((HERE / "pricing.json").read_text())
USAGE_GLOBS = ["usage.jsonl", "out/jac_*.calls.jsonl", "swebench/*/logs/*/byllm_usage.jsonl"]


def bare(m):
    return m.split("/", 1)[1] if "/" in m else m


def cost(model, pt, ct):
    p = PRICES.get(bare(model))
    return 0.0 if not p else (pt * p[0] + ct * p[2]) / 1e6


def usage(mode_dir):
    tot = collections.Counter(); by_site = collections.defaultdict(collections.Counter); models = collections.Counter()
    for pat in USAGE_GLOBS:
        for f in glob.glob(str(mode_dir / pat)):
            for line in open(f):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "prompt_tokens" not in r:
                    continue
                m = bare(r.get("model", "?"))
                tot["calls"] += 1; tot["pt"] += r["prompt_tokens"]; tot["ct"] += r["completion_tokens"]
                tot["cost"] += cost(m, r["prompt_tokens"], r["completion_tokens"])
                tot["escalated"] += int(r.get("route_reason") == "escalate")
                by_site[(r.get("scope") or "visit").rsplit(".", 1)[-1] or "visit"][m] += 1
                models[m] += 1
    return tot, by_site, models


def quality(arm, d):
    try:
        if arm == "meeting":
            runs = json.load(open(d / "scores.json"))
            n = len(runs) or 1
            q = {"completed": sum(r["completed"] for r in runs) / n,
                 "count in range": sum(bool(r["count_in_range"]) for r in runs) / n,
                 "pipeline consistent": sum(bool(r["pipeline_consistent"]) for r in runs) / n,
                 "malformed tasks": sum(r["malformed_tasks"] for r in runs)}
            js = [r["judge"] for r in runs if r.get("judge") and r["judge"].get("f1") is not None]
            if js:
                q["judge F1"] = sum(j["f1"] for j in js) / len(js)
                q["judge overall"] = sum(j.get("overall", 0) for j in js) / len(js)
            return q
        if arm == "factcheck":
            rs = [json.load(open(f)) for f in sorted(glob.glob(str(d / "out" / "jac_*.json")))]
            n = len(rs) or 1
            return {"claims": len(rs), "accuracy": sum(bool(r.get("correct")) for r in rs) / n,
                    "abstain": sum(r.get("verdict") == "NEED_MORE" for r in rs) / n,
                    "errors": sum(bool(r.get("error")) for r in rs),
                    "verdicts": " ".join(f"{r['verdict'][:3]}/{r['gold'][:3]}" for r in rs)}
        if arm == "ytnav":
            rep = json.load(open(d / "report.json"))["byllm"]
            m = rep["metrics"]
            q = {"questions": m["questions"], "completed": m["completed"], "routing acc": m["routing_accuracy"],
                 "retrieval hit": m["retrieval_hit_rate"], "answer parse": m["answer_parse_rate"], "errors": m["error_rate"]}
            js = rep.get("judge_scores") or {}
            if js:
                q["judge answer 1-5"] = sum(js.values()) / len(js)
            return q
        if arm == "codeagent":
            q = {}
            for f in glob.glob(str(d / "swebench" / "*" / "logs" / "*" / "result.json")):
                r = json.load(open(f)); q[Path(f).parent.name] = f"steps={r.get('steps')} err={'y' if r.get('error') else 'n'}"
            for f in glob.glob(str(d / "swebench" / "*" / "eval_results.jsonl")):
                for line in open(f):
                    r = json.loads(line); iid = r.get("instance_id", "?")
                    e = (r.get("entry") or {}).get(iid, r)
                    q[iid] = q.get(iid, "") + f" resolved={'yes' if e.get('resolved') else 'no'}"
            n = len(q); res = sum(v.endswith("resolved=yes") for v in q.values())
            return {"resolved": f"{res}/{n}", **q}
    except Exception as exc:
        return {"error": str(exc)[:80]}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--arms", nargs="+", default=["meeting", "factcheck", "ytnav", "codeagent"])
    args = ap.parse_args()
    out = []
    for arm in args.arms:
        base = Path(args.results) / arm
        if not (base / "eval").exists():
            continue
        out.append(f"## {arm}")
        rs = base / "router" / "summary.json"
        if rs.exists():
            s = json.loads(rs.read_text())
            out.append(f"Router: KNN trained on {s['train_requests']} requests (held-out oracle acc {s['router_oracle_accuracy']:.0%} on {s['test_requests']}); "
                       f"offline test picks {s['test_picks']}")
        out += ["", "| mode | LLM calls | prompt tok | completion tok | cost USD | escalated retries | quality |", "|---|---:|---:|---:|---:|---:|---|"]
        sites = []
        for mode in ("strong", "cheap", "routed"):
            d = base / "eval" / mode
            if not d.exists():
                continue
            tot, by_site, models = usage(d)
            q = "; ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in quality(arm, d).items())
            out.append(f"| {mode} | {tot['calls']} | {tot['pt']:,} | {tot['ct']:,} | {tot['cost']:.4f} | {tot['escalated']} | {q} |")
            sites.append((mode, by_site))
        out += ["", "Call sites (calls per model):"]
        for mode, by_site in sites:
            out.append(f"- **{mode}**: " + "; ".join(f"{site}: " + ", ".join(f"{m} x{n}" for m, n in ms.most_common()) for site, ms in sorted(by_site.items())))
        out.append("")
    text = "\n".join(out)
    (Path(args.results) / "REPORT_arms.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
