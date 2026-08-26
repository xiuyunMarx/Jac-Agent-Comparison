#!/usr/bin/env python3
"""Two figures: every implementation against the openai_sdk baseline.

    python3 plot_normalized.py --out-dir local-8 \
        ../swebench_bridge/results/local-8v1-byllm \
        ../swebench_bridge/results/local-8-langgraph \
        ../swebench_bridge/results/local-8-openai \
        ../swebench_bridge/results/local-8-nooa

    local-8/normalized_score.pdf   -- rates, normalised, higher is better
    local-8/normalized_tokens.pdf  -- tokens per instance, normalised, lower is cheaper

Both are grouped bars with the openai_sdk value of each group divided out, so
the baseline is 1.0 and the dashed line at 1.0 is the thing to read against --
the same shape as the RagGPT figures (image.png, TOKEN-USAGE.png at the repo
root), so the CodeAgent panels can sit beside them.

The score figure's groups are the three rates a graded run yields: resolved,
patch produced (non-empty), patch applied (non-empty and the harness could
apply it). All rates, so they normalise directly and higher is better.

The token figure's groups are the instances themselves, plus the mean: each
bar is that implementation's prompt+completion tokens on that instance over
openai_sdk's on the same instance. Nothing is inverted. A bar above 1.0 spent
more than the baseline did, exactly as in TOKEN-USAGE.png.

Reads what grade.py and run_agent.py wrote, through report.load_side, so the
numbers are the ones report.py prints and not a second computation of them.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "swebench_bridge"))

import report  # noqa: E402

# Framework -> (legend label, colour, hatch). jac/byLLM, framework and
# openai_sdk match the sibling figures; NOOA is the fourth arm and takes the
# fourth slot in the same visual language.
STYLE = {
    "byllm": ("jac/byLLM", "#d9534f", ""),
    "langgraph": ("framework", "#5cb85c", "//"),
    "openai": ("openai_sdk", "#4a6fb5", ".."),
    "nooa": ("NOOA", "#e6a23c", "xx"),
}
BASELINE = "openai"


def rc() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "font.size": 14,
        "axes.labelsize": 15,
        "axes.labelweight": "bold",
        "legend.fontsize": 14,
        "hatch.linewidth": 1.0,
        "pdf.fonttype": 42,
    })


def order_of(by_fw: dict) -> list[str]:
    return [fw for fw in STYLE if fw in by_fw] + sorted(set(by_fw) - set(STYLE))


def instance_tokens(side: dict) -> dict[str, int]:
    return {iid: int(r.get("prompt_tokens", 0) or 0) + int(r.get("completion_tokens", 0) or 0)
            for iid, r in side["runs"].items()}


def applied_ids(side: dict) -> set[str]:
    """Instances whose patch the harness applied: non-empty and not an apply failure.

    grade.py's eval_results.jsonl carries `patch_successfully_applied` per
    instance; the harness report only counts. Read the rows when present.
    """
    path = side["report_path"].parent / "eval_results.jsonl"
    if not path.exists():
        return set(side["resolved"])
    import json
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # One line per instance: {"instance_id", "entry": {iid: harness report}}.
        entry = (row.get("entry") or {}).get(row.get("instance_id"), row)
        if entry.get("patch_successfully_applied") and entry.get("patch_exists", True):
            out.add(row["instance_id"])
    return out


def grouped_bars(ax, groups: list[str], order: list[str], scores: dict[str, list[float]]):
    from matplotlib.patches import Patch
    n_fw, n_g = len(order), len(groups)
    width = 0.8 / n_fw
    handles = []
    for i, fw in enumerate(order):
        label, colour, hatch = STYLE.get(fw, (fw, "#999999", ""))
        xs = [g + (i - (n_fw - 1) / 2) * width for g in range(n_g)]
        ax.bar(xs, scores[fw], width=width * 0.96, color=colour, hatch=hatch,
               edgecolor="black", linewidth=1.2, zorder=3)
        handles.append(Patch(facecolor=colour, hatch=hatch, edgecolor="black",
                             linewidth=1.2, label=label))
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.6, zorder=2)
    ax.set_xticks(range(n_g))
    ax.set_xticklabels(groups, rotation=20, ha="right")
    top = max((v for vs in scores.values() for v in vs), default=1.0)
    ax.set_ylim(0, max(1.5, top * 1.12))
    ax.yaxis.grid(True, linestyle="--", color="#bbbbbb", zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=n_fw, frameon=True, edgecolor="#666666", fancybox=False,
              handlelength=2.2, borderpad=0.5)


def finish(fig, out: Path, footnote: str) -> None:
    import matplotlib.pyplot as plt
    fig.text(0.01, 0.006, "\n".join(textwrap.wrap(footnote, 150)), fontsize=8.5,
             color="#333333", ha="left", va="bottom", linespacing=1.4)
    fig.tight_layout(rect=(0.005, 0.06, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def score_figure(by_fw: dict[str, dict], out: Path, note: str) -> dict:
    import matplotlib.pyplot as plt
    order = order_of(by_fw)
    groups = ["Resolved", "Patch produced", "Patch applied"]
    rates = {}
    for fw, side in by_fw.items():
        n = side["submitted"] or 1
        produced = n - side["empty"] - side["errored"]
        rates[fw] = [len(side["resolved"]) / n, produced / n, len(applied_ids(side)) / n]
    base = rates[BASELINE]
    scores = {fw: [(v / b) if b else 0.0 for v, b in zip(rates[fw], base)] for fw in order}
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    grouped_bars(ax, groups, order, scores)
    ax.set_ylabel("Normalized Evaluation Score", labelpad=10)
    n = by_fw[BASELINE]["submitted"]
    finish(fig, out, f"CodeAgent, SWE-bench Lite, {n} leaderboard-easy instances. "
           f"1.0 = the openai_sdk baseline; higher is better. Resolved = FAIL_TO_PASS pass and "
           f"PASS_TO_PASS intact; patch produced = non-empty; patch applied = the harness "
           f"could apply it.{(' ' + note) if note else ''}")
    return {fw: dict(zip(groups, scores[fw])) for fw in order}


def token_figure(by_fw: dict[str, dict], out: Path, note: str) -> dict:
    import matplotlib.pyplot as plt
    order = order_of(by_fw)
    per = {fw: instance_tokens(side) for fw, side in by_fw.items()}
    ids = sorted(set(per[BASELINE]) & set.intersection(*(set(p) for p in per.values())))
    short = [i.split("__", 1)[1] for i in ids]
    groups = short + ["mean of all"]
    scores = {}
    for fw in order:
        s = [per[fw][i] / per[BASELINE][i] if per[BASELINE][i] else 0.0 for i in ids]
        mean_fw = sum(per[fw][i] for i in ids) / (len(ids) or 1)
        mean_b = sum(per[BASELINE][i] for i in ids) / (len(ids) or 1)
        s.append(mean_fw / mean_b if mean_b else 0.0)
        scores[fw] = s
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    grouped_bars(ax, groups, order, scores)
    ax.set_ylabel("Normalized Token Usage", labelpad=10)
    finish(fig, out, f"CodeAgent, SWE-bench Lite, {len(ids)} leaderboard-easy instances. "
           f"Prompt + completion tokens per instance over openai_sdk's on the same instance. "
           f"1.0 = the openai_sdk baseline; lower is cheaper.{(' ' + note) if note else ''}")
    return {fw: dict(zip(groups, scores[fw])) for fw in order}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="graded run directories")
    p.add_argument("--out-dir", type=Path, default=HERE)
    p.add_argument("--note", default="", help="appended to both footnotes, e.g. the model")
    args = p.parse_args(argv)

    rc()
    by_fw = {}
    for d in args.runs:
        side = report.load_side(d)
        by_fw[side["framework"]] = side
    if BASELINE not in by_fw:
        raise SystemExit(f"no {BASELINE} run among the sides; it is the baseline")

    s = score_figure(by_fw, args.out_dir / "normalized_score.pdf", args.note)
    t = token_figure(by_fw, args.out_dir / "normalized_tokens.pdf", args.note)
    print(args.out_dir / "normalized_score.pdf")
    for fw, row in s.items():
        print(f"  {STYLE.get(fw, (fw,))[0]:12s} " + "  ".join(f"{k}={v:.2f}" for k, v in row.items()))
    print(args.out_dir / "normalized_tokens.pdf")
    for fw, row in t.items():
        print(f"  {STYLE.get(fw, (fw,))[0]:12s} mean={row['mean of all']:.2f}  " +
              " ".join(f"{k.split('-')[-1]}={v:.1f}" for k, v in row.items() if k != "mean of all"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
