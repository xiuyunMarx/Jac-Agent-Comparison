"""Aggregate judged results into results/report.md and CSVs.

Run:  /home/xiaoyu/miniconda3/envs/jaseci/bin/python report.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

# Pricing, USD per 1M tokens (input, output). Defaults to gpt-4.1-mini's, the
# model this eval was written against. A locally served model costs nothing per
# token, and reporting it at OpenAI's rates would invent a dollar figure -- so
# the runner sets both to 0 and the $/1k column reads as the zero it is.
PRICE_IN = float(os.environ.get("BENCH_PRICE_IN", "0.40"))
PRICE_OUT = float(os.environ.get("BENCH_PRICE_OUT", "1.60"))

BOOT_ITERS = 2000


def bootstrap_ci(per_item: np.ndarray, iters: int = BOOT_ITERS, seed: int = 0) -> tuple[float, float]:
    if len(per_item) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_item), size=(iters, len(per_item)))
    means = per_item[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def fmt_ci(mean: float, lo: float, hi: float, pct: bool = True) -> str:
    if pct:
        return f"{mean:.1%} [{lo:.1%}, {hi:.1%}]"
    return f"{mean:.2f} [{lo:.2f}, {hi:.2f}]"


def main() -> None:
    df = pd.DataFrame(common.read_jsonl(common.JUDGED_PATH))
    if df.empty:
        raise SystemExit("no judged rows — run score.py first")
    systems = [s for s in common.SYSTEMS if s in set(df["system"])]
    common.RESULTS_DIR.mkdir(exist_ok=True)
    lines = ["# Jac-GPT three-system eval report", ""]
    n_items = df["item_id"].nunique()
    n_reps = df["repeat"].max()
    lines.append(f"{n_items} items, {int(n_reps)} repeats, systems: {', '.join(systems)}.")
    lines.append("")

    # ---------------- headline table
    rows = []
    for sysname in systems:
        d = df[df["system"] == sysname]
        code = d[d["category"].isin(["coding", "debugging"])]
        per_item_routing = d.groupby("item_id")["routing_correct"].mean().to_numpy()
        lo, hi = bootstrap_ci(per_item_routing)
        doc = d[d["category"].isin(["rag_qa", "coding", "debugging"])]
        rows.append({
            "system": sysname,
            "routing_acc": fmt_ci(d["routing_correct"].mean(), lo, hi),
            "judge_score(1-5)": round(doc["judge_score"].dropna().mean(), 2),
            "compile_rate": f"{(code['compiles'] == True).mean():.1%}",
            "appropriate_rate": f"{d['appropriate'].dropna().mean():.1%}"
                                 if d["appropriate"].notna().any() else "-",
            "error_rate": f"{d['failed'].mean():.1%}",
            "llm_calls/turn": round(d["llm_calls"].mean(), 2),
            "prompt_tok/turn": int(d["prompt_tokens"].mean()),
            "completion_tok/turn": int(d["completion_tokens"].mean()),
            "$/1k turns": round((d["prompt_tokens"].mean() * PRICE_IN
                                 + d["completion_tokens"].mean() * PRICE_OUT) / 1e6 * 1000, 2),
            "latency_p50_ms": int(d["latency_ms"].median()),
            "latency_p95_ms": int(d["latency_ms"].quantile(0.95)),
        })
    headline = pd.DataFrame(rows)
    headline.to_csv(common.RESULTS_DIR / "summary_by_system.csv", index=False)
    lines += ["## Headline (all turns)", "", headline.to_markdown(index=False), ""]

    # ---------------- per-category tables
    for metric, col, kind in [("Routing accuracy", "routing_correct", "pct"),
                              ("Mean judge score (1-5)", "judge_score", "num"),
                              ("Total tokens / turn", "total_tokens", "int"),
                              ("LLM calls / turn", "llm_calls", "num")]:
        pivot = df.pivot_table(index="category", columns="system", values=col, aggfunc="mean")
        pivot = pivot.reindex(index=[c for c in common.CATEGORIES if c in pivot.index],
                              columns=systems)
        if kind == "pct":
            shown = pivot.map(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
        elif kind == "int":
            shown = pivot.map(lambda v: f"{v:,.0f}" if pd.notna(v) else "-")
        else:
            shown = pivot.map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
        lines += [f"## {metric} by category", "", shown.to_markdown(), ""]
    df.pivot_table(index="category", columns="system", values="total_tokens",
                   aggfunc="mean").to_csv(common.RESULTS_DIR / "tokens_by_category.csv")

    # ---------------- multi-turn turn-2 routing (the discriminative subset)
    mt2 = df[(df["category"] == "multi_turn") & (df["turn"] == 1)]
    if not mt2.empty:
        t = mt2.groupby("system")["routing_correct"].mean().reindex(systems)
        lines += ["## Multi-turn: turn-2 routing accuracy (context-dependent follow-ups)", "",
                  t.map(lambda v: f"{v:.1%}").to_markdown(), ""]

    # ---------------- confusion matrices
    lines += ["## Routing confusion (gold rows x routed columns, all repeats)", ""]
    for sysname in systems:
        d = df[(df["system"] == sysname) & (~df["failed"])]
        conf = pd.crosstab(d["gold_agent"], d["routed_agent"])
        conf = conf.reindex(index=[a for a in common.AGENTS if a in conf.index])
        conf.to_csv(common.RESULTS_DIR / f"confusion_{sysname}.csv")
        lines += [f"### {sysname}", "", conf.to_markdown(), ""]

    # ---------------- pairwise significance (paired per item)
    lines += ["## Pairwise comparisons (Wilcoxon signed-rank over per-item means)", "",
              "| metric | pair | delta (A-B) | p-value |", "|---|---|---|---|"]
    doc_mask = df["category"].isin(["rag_qa", "coding", "debugging"])
    metric_frames = {
        "judge_score": df[doc_mask].dropna(subset=["judge_score"]),
        "total_tokens": df,
        "latency_ms": df,
        "routing_correct": df,
    }
    for metric, frame in metric_frames.items():
        per = frame.groupby(["system", "item_id"])[metric].mean().unstack("system")
        for i, a in enumerate(systems):
            for b in systems[i + 1:]:
                if a not in per.columns or b not in per.columns:
                    continue
                paired = per[[a, b]].dropna()
                if len(paired) < 10:
                    continue
                delta = float(paired[a].mean() - paired[b].mean())
                diffs = paired[a] - paired[b]
                if np.allclose(diffs, 0):
                    p = 1.0
                else:
                    p = float(stats.wilcoxon(paired[a], paired[b]).pvalue)
                lines.append(f"| {metric} | {a} vs {b} | {delta:+.2f} | {p:.4f} |")
    lines.append("")

    # ---------------- notes
    lines += [
        "## Reading notes",
        "",
        "- The three systems share verbatim agent prompts and RAG config; differences trace to "
        "the router mechanism and framework overhead (byllm vs LangGraph serialization).",
        "- Known asymmetries by design: Jac-Rag-GPT routes at byllm's default temperature 0.7 "
        "(others at 0); the LangGraph router never sees chat history; fallback agents differ "
        "(LangGraph -> RagChat, ByllmRouter -> OffTopicChat).",
        f"- $/1k turns assumes gpt-4.1-mini at ${PRICE_IN}/M input, ${PRICE_OUT}/M output.",
        "- multi_turn turn-2 accuracy isolates history-dependent routing.",
    ]

    out = common.RESULTS_DIR / "report.md"
    out.write_text("\n".join(lines))
    print(f"report -> {out}")
    print(headline.to_string(index=False))


if __name__ == "__main__":
    main()
