"""Score eval/out: task performance and token usage per arm, as a markdown report.

    python eval/score.py [--out eval/out] [--report eval/out/report.md]

Task performance: HoVer is a binary task, so accuracy counts SUPPORTED /
NOT_SUPPORTED against gold and treats NEED_MORE (the round cap hit) and any
failed run as wrong. Also reported: accuracy on decided claims only, the
abstain rate, and accuracy by hop count.
Token usage: per-claim mean and median of model calls, prompt tokens,
completion tokens, correction retries, and wall time.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["jac", "openai", "langgraph"]
NAMES = {"jac": "Jac (byLLM)", "openai": "OpenAI SDK", "langgraph": "LangGraph"}


def load(out: Path) -> dict[str, list[dict]]:
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(out.glob("*_???.json")):
        r = json.loads(p.read_text())
        by_arm[r["arm"]].append(r)
    return by_arm


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "n/a"


def mean_med(vals: list[float]) -> str:
    vals = [v for v in vals if v is not None]
    return f"{st.mean(vals):,.0f} / {st.median(vals):,.0f}" if vals else "n/a"


def report(by_arm: dict[str, list[dict]]) -> str:
    arms = [a for a in ORDER if a in by_arm] + [a for a in by_arm if a not in ORDER]
    lines = ["# FactCheck 3-way comparison (GLM-5.2, HoVer dev claims)", ""]
    n_claims = max(len(v) for v in by_arm.values())
    lines.append(f"{n_claims} claims per arm; knobs: FC_SCOUTS=2, FC_ROUNDS=FC_MIN_ROUNDS=6 (defaults). "
                 "NEED_MORE = round cap reached without a decision, scored as wrong.")
    lines.append("")

    lines.append("## Task performance")
    lines.append("")
    lines.append("| arm | runs | failed | accuracy | decided | acc. on decided | abstain (NEED_MORE) | SUPPORTED acc. | NOT_SUPPORTED acc. |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        rs = by_arm[a]
        n = len(rs)
        failed = sum(1 for r in rs if r["error"])
        correct = sum(1 for r in rs if r["correct"])
        decided = [r for r in rs if r["verdict"] in ("SUPPORTED", "NOT_SUPPORTED")]
        dec_correct = sum(1 for r in decided if r["correct"])
        abstain = sum(1 for r in rs if r["verdict"] == "NEED_MORE")
        sup = [r for r in rs if r["gold"] == "SUPPORTED"]
        nsup = [r for r in rs if r["gold"] == "NOT_SUPPORTED"]
        lines.append(
            f"| {NAMES.get(a, a)} | {n} | {failed} | **{pct(correct, n)}** ({correct}/{n}) | {len(decided)} | "
            f"{pct(dec_correct, len(decided))} | {pct(abstain, n)} | "
            f"{pct(sum(r['correct'] for r in sup), len(sup))} | {pct(sum(r['correct'] for r in nsup), len(nsup))} |"
        )
    lines.append("")

    hops = sorted({r["hops"] for rs in by_arm.values() for r in rs})
    lines.append("### Accuracy by hop count")
    lines.append("")
    lines.append("| arm | " + " | ".join(f"{h} hops" for h in hops) + " |")
    lines.append("|---|" + "---|" * len(hops))
    for a in arms:
        cells = []
        for h in hops:
            rs = [r for r in by_arm[a] if r["hops"] == h]
            cells.append(f"{pct(sum(r['correct'] for r in rs), len(rs))} ({sum(r['correct'] for r in rs)}/{len(rs)})")
        lines.append(f"| {NAMES.get(a, a)} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### Verdict distribution")
    lines.append("")
    lines.append("| arm | SUPPORTED | NOT_SUPPORTED | NEED_MORE | none (failed) |")
    lines.append("|---|---|---|---|---|")
    for a in arms:
        rs = by_arm[a]
        c = lambda v: sum(1 for r in rs if r["verdict"] == v)
        lines.append(f"| {NAMES.get(a, a)} | {c('SUPPORTED')} | {c('NOT_SUPPORTED')} | {c('NEED_MORE')} | {c(None)} |")
    lines.append("")

    lines.append("## Token usage (per claim, mean / median)")
    lines.append("")
    lines.append("| arm | model calls | retries | prompt tokens | completion tokens | total tokens | wall s | total prompt tokens (all claims) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for a in arms:
        rs = [r for r in by_arm[a] if r["calls"]]
        lines.append(
            f"| {NAMES.get(a, a)} | {mean_med([r['calls'] for r in rs])} | {mean_med([r['retries'] for r in rs])} | "
            f"{mean_med([r['prompt_tokens'] for r in rs])} | {mean_med([r['completion_tokens'] for r in rs])} | "
            f"{mean_med([r['prompt_tokens'] + r['completion_tokens'] for r in rs])} | "
            f"{mean_med([r['wall_s'] for r in rs])} | {sum(r['prompt_tokens'] for r in rs):,} |"
        )
    lines.append("")
    lines.append("Retries: for the Python arms, correction turns after an unparseable reply (traced as `*/retry`); "
                 "byLLM's own correction calls are not tagged in its usage log and show up only in the call count.")
    lines.append("")

    lines.append("## Per-claim verdicts")
    lines.append("")
    lines.append("| # | hops | gold | " + " | ".join(NAMES.get(a, a) for a in arms) + " | prompt tokens (" + "/".join(arms) + ") |")
    lines.append("|---|---|---|" + "---|" * (len(arms) + 1))
    idxs = sorted({r["idx"] for rs in by_arm.values() for r in rs})
    for i in idxs:
        row = {a: next((r for r in by_arm[a] if r["idx"] == i), None) for a in arms}
        gold = next(r["gold"] for r in row.values() if r)
        hops_i = next(r["hops"] for r in row.values() if r)

        def cell(r):
            if r is None:
                return "-"
            if r["error"]:
                return f"FAILED"
            return ("✓ " if r["correct"] else "✗ ") + (r["verdict"] or "none")

        toks = "/".join(f"{row[a]['prompt_tokens'] // 1000}k" if row[a] else "-" for a in arms)
        lines.append(f"| {i} | {hops_i} | {gold} | " + " | ".join(cell(row[a]) for a in arms) + f" | {toks} |")
    lines.append("")

    failed = [(a, r) for a in arms for r in by_arm[a] if r["error"]]
    if failed:
        lines.append("## Failed runs")
        lines.append("")
        for a, r in failed:
            lines.append(f"- {NAMES.get(a, a)} #{r['idx']:03d} (exit {r['exit']}): {r['error']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "eval/out"))
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    by_arm = load(out)
    if not by_arm:
        raise SystemExit(f"no results in {out}")
    text = report(by_arm)
    Path(args.report or out / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
