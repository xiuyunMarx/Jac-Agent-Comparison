"""Summarise eval/out/*.json into eval/out/report.md (also printed).

    python eval/score.py

Per arm: accuracy, runs that crashed, runs that hit the step cap, and per-task
means of model calls, tokens, tool calls and wall time; then a task-by-arm
verdict table. A crash counts as a failed task.

Hitting the step cap (16 model calls) ends differently in each framework, so it
is detected per arm: byLLM makes a 17th, forced-answer call; LangGraph's prebuilt agent
answers "Sorry, need more steps..."; NOOA's CodeAct raises after
max_iterations; the SDK arm raises "no answer after 16 steps".
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
ARMS = ["jac", "openai", "langgraph", "nooa"]
MAX_STEPS = 16   # every arm's step budget


def hit_cap(r: dict) -> bool:
    diagnostics = r.get("diagnostics") or ""
    return {
        "jac": r["calls"] > MAX_STEPS,
        "openai": "no answer after" in diagnostics,
        "langgraph": (r.get("answer") or "").startswith("Sorry, need more steps"),
        "nooa": "max_iterations" in diagnostics,
    }[r["arm"]]


def mean(rows: list[dict], key: str) -> str:
    return f"{statistics.mean(r[key] for r in rows):,.1f}" if rows else "-"


def median(rows: list[dict], key: str) -> str:
    return f"{statistics.median(r[key] for r in rows):,.0f}" if rows else "-"


def main() -> None:
    results = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    by_arm = {arm: [r for r in results if r["arm"] == arm] for arm in ARMS}
    lines = [
        "# ReAct loop on BFCL v4 web_search",
        "",
        "| arm | runs | passed | accuracy | crashed | step cap | calls | prompt tok (mean / median) | completion tok | tool calls | wall s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, rows in by_arm.items():
        if not rows:
            continue
        passed = sum(r["verdict"] == "PASSED" for r in rows)
        lines.append(
            f"| {arm} | {len(rows)} | {passed} | {passed / len(rows):.1%} | {sum(bool(r['error']) for r in rows)} "
            f"| {sum(hit_cap(r) for r in rows)} | {mean(rows, 'calls')} "
            f"| {mean(rows, 'prompt_tokens')} / {median(rows, 'prompt_tokens')} | {mean(rows, 'completion_tokens')} "
            f"| {mean(rows, 'tool_calls')} | {mean(rows, 'wall_s')} |"
        )

    arms = [arm for arm in ARMS if by_arm[arm]]
    verdicts = {(r["arm"], r["task"]): r["verdict"] or "ERROR" for r in results}
    tasks = sorted({r["task"] for r in results}, key=lambda t: int(t.rsplit("_", 1)[-1]))
    lines += ["", "## Per task", "", "| task | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for task in tasks:
        lines.append(f"| {task} | " + " | ".join(verdicts.get((arm, task), "") for arm in arms) + " |")

    report = "\n".join(lines) + "\n"
    (OUT / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
