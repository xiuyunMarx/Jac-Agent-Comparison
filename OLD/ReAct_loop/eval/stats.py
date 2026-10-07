"""Equivalence test of every arm against the Jac arm, paired by task: eval/out/*.json -> eval/out/stats.md.

    python eval/stats.py

Only tasks every arm finished without an error are used. For each arm vs jac:
- accuracy: mean per-task difference in pass rate (other - jac), in percentage points;
- tokens: geometric-mean ratio of total tokens (prompt + completion), other / jac.
Confidence intervals are paired Student-t intervals over tasks (on log ratios for tokens). An arm
is "equivalent" to jac when the 90% CI lies inside the margin (TOST at alpha 0.05), and
"different" when the 95% CI excludes no difference; a result can be both, or neither
("inconclusive").
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
ARMS = ["jac", "openai", "langgraph", "nooa"]
BASE = "jac"
ACC_MARGIN = 0.15               # equivalent if the pass-rate difference stays within ±15 points
TOKEN_MARGIN = (0.70, 1.30)     # equivalent if the token ratio stays within 0.70-1.30


def t_quantile(p: float, df: int) -> float:
    # Cornish-Fisher expansion of Student's t around the normal quantile (error < 0.002 for df >= 4)
    z = statistics.NormalDist().inv_cdf(p)
    g = [
        (z**3 + z) / 4,
        (5 * z**5 + 16 * z**3 + 3 * z) / 96,
        (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384,
        (79 * z**9 + 776 * z**7 + 1482 * z**5 - 1920 * z**3 - 945 * z) / 92160,
    ]
    return z + sum(gi / df ** (i + 1) for i, gi in enumerate(g))


def ci(values: list[float], level: float) -> tuple[float, float]:
    n = len(values)
    half = t_quantile(1 - (1 - level) / 2, n - 1) * statistics.stdev(values) / math.sqrt(n)
    return statistics.mean(values) - half, statistics.mean(values) + half


def verdict(ci90: tuple[float, float], ci95: tuple[float, float], lo: float, hi: float, null: float) -> str:
    equivalent = lo < ci90[0] and ci90[1] < hi
    different = not (ci95[0] <= null <= ci95[1])
    if equivalent and different:
        return "different, but within the margin"
    return "equivalent" if equivalent else "different" if different else "inconclusive"


def main() -> None:
    results = {(r["arm"], r["task"]): r for r in (json.loads(p.read_text()) for p in OUT.glob("*.json"))}
    tasks = sorted(
        {t for (_, t) in results if all((a, t) in results and not results[(a, t)]["error"] for a in ARMS)},
        key=lambda t: int(t.rsplit("_", 1)[-1]),
    )
    total = {k: r["prompt_tokens"] + r["completion_tokens"] for k, r in results.items()}
    passed = {k: float(r["verdict"] == "PASSED") for k, r in results.items()}

    lines = [
        f"# Equivalence vs {BASE} ({len(tasks)} tasks, paired)",
        "",
        f"Margins: accuracy ±{ACC_MARGIN:.0%} points, token ratio {TOKEN_MARGIN[0]:.2f}-{TOKEN_MARGIN[1]:.2f}. "
        f"Equivalent = 90% CI inside the margin; different = 95% CI excludes no difference.",
        "",
        "| arm | accuracy | mean tokens (prompt + completion) |",
        "|---|---|---|",
    ]
    for arm in ARMS:
        acc = statistics.mean(passed[(arm, t)] for t in tasks)
        lines.append(f"| {arm} | {acc:.0%} | {statistics.mean(total[(arm, t)] for t in tasks):,.0f} |")

    lines += [
        "",
        f"| arm vs {BASE} | accuracy diff (points) | 90% CI | 95% CI | verdict "
        f"| token ratio | 90% CI | 95% CI | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        if arm == BASE:
            continue
        diffs = [passed[(arm, t)] - passed[(BASE, t)] for t in tasks]
        a90, a95 = ci(diffs, 0.90), ci(diffs, 0.95)
        logs = [math.log(total[(arm, t)] / total[(BASE, t)]) for t in tasks]
        t90, t95 = (tuple(math.exp(x) for x in ci(logs, level)) for level in (0.90, 0.95))
        lines.append(
            f"| {arm} | {statistics.mean(diffs) * 100:+.1f} | [{a90[0] * 100:+.1f}, {a90[1] * 100:+.1f}] "
            f"| [{a95[0] * 100:+.1f}, {a95[1] * 100:+.1f}] | {verdict(a90, a95, -ACC_MARGIN, ACC_MARGIN, 0.0)} "
            f"| x{math.exp(statistics.mean(logs)):.2f} | [x{t90[0]:.2f}, x{t90[1]:.2f}] | [x{t95[0]:.2f}, x{t95[1]:.2f}] "
            f"| {verdict(t90, t95, *TOKEN_MARGIN, 1.0)} |"
        )

    report = "\n".join(lines) + "\n"
    (OUT / "stats.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
