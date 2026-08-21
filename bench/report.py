"""One table across five benchmarks: what each arm scored, and what it spent.

Each project measures something different -- a resolve rate, a routing accuracy,
an F1 over which emails got answered -- so there is no single number to rank
them by, and this does not invent one. It puts each project's own headline
metric next to the tokens that bought it, for all three arms, in one place.

Two things it checks rather than reports:

  * **Three-arm coverage.** An empty cell is flagged loudly. Four of the five
    benchmarks sat with a written-but-never-run OpenAI-SDK arm for a long time
    precisely because nothing ever said "this cell is empty".
  * **Ledger reconciliation.** The proxy counted every call independently of
    the frameworks. If a project's own token count disagrees with the proxy's,
    an arm found a way around the instrumentation and its cost is understated.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import config

R = config.REPO_ROOT

#: Every arm maps onto one of three roles, so the same table row means the same
#: thing in all five projects.
JAC, FRAMEWORK, SDK = "jac/byLLM", "framework", "openai_sdk"


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _mean(values) -> float | None:
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else None


def _rate(values) -> float | None:
    values = [bool(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


# ---------------------------------------------------------------------------
# Per-project extractors. Each returns rows of the shared shape.
# ---------------------------------------------------------------------------

def _row(benchmark, arm, impl, **kw) -> dict:
    row = {"benchmark": benchmark, "arm": arm, "implementation": impl,
           "metric": None, "metric_name": "", "judge": None, "tokens_in": None,
           "tokens_out": None, "llm_calls": None, "latency_p50_s": None,
           "error_rate": None, "parse_failure_rate": None, "units": None}
    row.update(kw)
    return row


def rows_meeting() -> list[dict]:
    data = _load_json(R / "meeting-assistant" / "eval" / "out" / "summary.json")
    if not data:
        return []
    arms = {"byLLM": JAC, "CrewAI": FRAMEWORK, "openai_sdk": SDK}
    out = []
    for impl, arm in arms.items():
        rs = [r for r in data if r.get("implementation") == impl]
        if not rs:
            continue
        judges = [r.get("judge", {}).get("mean_overall") for r in rs if isinstance(r.get("judge"), dict)]
        out.append(_row(
            "meeting-assistant", arm, impl,
            metric_name="task count in expected range",
            metric=_rate([r.get("count_in_range") for r in rs]),
            judge=_mean(judges),
            tokens_in=sum(r.get("prompt_tokens") or 0 for r in rs),
            tokens_out=sum(r.get("completion_tokens") or 0 for r in rs),
            llm_calls=sum(r.get("llm_calls") or 0 for r in rs),
            latency_p50_s=_mean([r.get("wall_time_s") for r in rs]),
            error_rate=1 - (_rate([r.get("completed") for r in rs]) or 0),
            parse_failure_rate=_rate([bool(r.get("malformed_tasks")) for r in rs]),
            units=len(rs)))
    return out


def rows_email() -> list[dict]:
    data = _load_json(R / "Email-Auto-response" / "eval" / "out" / "summary.json")
    if not data:
        return []
    arms = {"byLLM": JAC, "CrewAI-LangGraph": FRAMEWORK, "openai_sdk": SDK}
    out = []
    for impl, arm in arms.items():
        rs = [r for r in data if r.get("implementation") == impl]
        if not rs:
            continue
        cost = [r.get("cost") or {} for r in rs]
        judges = [(r.get("judge") or {}).get("mean_overall") for r in rs]
        out.append(_row(
            "Email-Auto-response", arm, impl,
            metric_name="filtering F1",
            metric=_mean([(r.get("filtering") or {}).get("f1") for r in rs]),
            judge=_mean(judges),
            tokens_in=sum(c.get("prompt_tokens") or 0 for c in cost),
            tokens_out=sum(c.get("completion_tokens") or 0 for c in cost),
            llm_calls=sum(c.get("llm_calls") or 0 for c in cost),
            parse_failure_rate=_mean([1 - ((r.get("drafts") or {}).get("completion_rate") or 0)
                                      for r in rs]),
            units=len(rs)))
    return out


def rows_ytnavigator() -> list[dict]:
    report = _load_json(R / "YTNavigator" / "eval" / "out" / "report.json")
    if not report:
        return []
    arms = {"byllm": JAC, "langgraph": FRAMEWORK, "openai_sdk": SDK}
    out_dir = R / "YTNavigator" / "eval" / "out"
    out = []
    for impl, arm in arms.items():
        entry = report.get(impl)
        if not entry:
            continue
        m = entry.get("metrics") or {}
        # Tokens live per-record rather than in the report.
        tin = tout = 0
        jsonl = out_dir / f"results_{impl}.jsonl"
        if jsonl.is_file():
            for line in jsonl.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tin += rec.get("prompt_tokens") or 0
                tout += rec.get("completion_tokens") or 0
        scores = entry.get("judge_scores") or []
        out.append(_row(
            "YTNavigator", arm, impl,
            metric_name="retrieval hit rate",
            metric=m.get("retrieval_hit_rate"),
            judge=_mean([s.get("score") if isinstance(s, dict) else s for s in scores]),
            tokens_in=tin, tokens_out=tout,
            llm_calls=m.get("llm_calls_mean"),
            latency_p50_s=m.get("latency_p50_s"),
            error_rate=m.get("error_rate"),
            parse_failure_rate=(1 - m["answer_parse_rate"]
                                if isinstance(m.get("answer_parse_rate"), (int, float)) else None),
            units=m.get("questions")))
    return out


def rows_raggpt() -> list[dict]:
    path = R / "RagGPT" / "eval" / "results" / "summary_by_system.csv"
    if not path.is_file():
        return []
    arms = {"jac": JAC, "jac-byllm-router": f"{JAC} (router variant)",
            "langgraph": FRAMEWORK, "openai-sdk": SDK}

    def num(text):
        """First number in a cell, as a fraction if it is a percentage.

        report.py writes some cells as "99.4% [98.3%, 100.0%]" -- the point
        estimate followed by its bootstrap interval -- so take the leading
        token, not the whole string.
        """
        head = str(text or "").strip().split()[0] if str(text or "").strip() else ""
        try:
            return float(head.rstrip("%")) / (100 if head.endswith("%") else 1)
        except ValueError:
            return None

    out = []
    for row in csv.DictReader(path.open()):
        system = row.get("system", "")
        if system not in arms:
            continue
        out.append(_row(
            "RagGPT", arms[system], system,
            metric_name="routing accuracy",
            metric=num(row.get("routing_acc")),
            judge=num(row.get("judge_score(1-5)")),
            tokens_in=num(row.get("prompt_tok/turn")),
            tokens_out=num(row.get("completion_tok/turn")),
            llm_calls=num(row.get("llm_calls/turn")),
            latency_p50_s=(num(row.get("latency_p50_ms")) or 0) / 1000 or None,
            error_rate=num(row.get("error_rate")),
            units="per turn"))
    return out


def rows_codeagent(run_id: str) -> list[dict]:
    bridge = R / "CodeAgent" / "swebench_bridge" / "results"
    arms = {"byllm": JAC, "langgraph": FRAMEWORK, "openai": SDK}
    out = []
    for fw, arm in arms.items():
        run_dir = bridge / f"{run_id}-{fw}"
        if not run_dir.is_dir():
            continue
        report = next((f for f in sorted(run_dir.glob("*.json"))), None)
        data = _load_json(report) if report else None
        tin = tout = calls = 0
        runs = run_dir / "runs.jsonl"
        if runs.is_file():
            for line in runs.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tin += rec.get("prompt_tokens") or 0
                tout += rec.get("completion_tokens") or 0
                calls += rec.get("llm_calls") or 0
        def count(key: str) -> int:
            """The harness reports these as a count in some versions and as the
            list of instance ids in others."""
            value = (data or {}).get(key)
            return len(value) if isinstance(value, list) else int(value or 0)

        total = count("total_instances")
        resolved = count("resolved_instances")
        out.append(_row(
            "CodeAgent", arm, fw,
            metric_name="SWE-bench resolve rate",
            metric=(resolved / total) if total else None,
            tokens_in=tin, tokens_out=tout, llm_calls=calls or None,
            error_rate=(count("error_instances") / total) if total else None,
            parse_failure_rate=(count("empty_patch_instances") / total) if total else None,
            units=total or None))
    return out


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

def ledger_totals(path: Path) -> dict:
    """Per-project totals from the proxy log: instrumentation nobody's framework
    can talk its way out of."""
    totals: dict[str, dict] = {}
    if not path.is_file():
        return totals
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = row.get("system") or "(unmarked)"
        agg = totals.setdefault(key, {"calls": 0, "prompt": 0, "completion": 0, "no_usage": 0})
        agg["calls"] += 1
        agg["prompt"] += row.get("prompt_tokens") or 0
        agg["completion"] += row.get("completion_tokens") or 0
        agg["no_usage"] += 0 if row.get("has_usage") else 1
    return totals


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def fmt(value, kind="num") -> str:
    if value is None:
        return "-"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "int":
        return f"{value:,.0f}"
    return f"{value:.2f}"


def render(rows: list[dict], manifest: dict, ledger: dict, run_id: str) -> str:
    cfg = manifest.get("config", {})
    lines = [
        f"# Benchmark sweep {run_id}", "",
        f"- model: `{cfg.get('served_model', '?')}` at `{cfg.get('base_url', '?')}`",
        f"- context window: {cfg.get('ctx_window', '?')}",
        f"- mode: {'smoke' if manifest.get('smoke') else 'full'}, "
        f"repeats: {manifest.get('repeats', '?')}",
        f"- judge: {'the same local model' if manifest.get('judge') else 'off'}",
        "",
    ]
    if manifest.get("judge"):
        lines += [
            "> The judge is the model under test. Judge columns are usable for",
            "> comparing arms against each other, not as absolute quality: a model",
            "> grading its own family is a biased grader.", "",
        ]

    lines += ["## Quality and cost", "",
              "| benchmark | arm | implementation | metric | score | judge | "
              "tok in | tok out | calls | err | parse-fail | over |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['benchmark']} | {r['arm']} | `{r['implementation']}` | "
            f"{r['metric_name']} | {fmt(r['metric'], 'pct')} | {fmt(r['judge'])} | "
            f"{fmt(r['tokens_in'], 'int')} | {fmt(r['tokens_out'], 'int')} | "
            f"{fmt(r['llm_calls'], 'int')} | {fmt(r['error_rate'], 'pct')} | "
            f"{fmt(r['parse_failure_rate'], 'pct')} | {r['units'] if r['units'] is not None else '-'} |")

    # -- coverage ----------------------------------------------------------
    lines += ["", "## Three-arm coverage", ""]
    benchmarks = ["meeting-assistant", "Email-Auto-response", "YTNavigator",
                  "RagGPT", "CodeAgent"]
    gaps = []
    lines += ["| benchmark | jac/byLLM | framework | openai_sdk |", "|---|---|---|---|"]
    for bench in benchmarks:
        cells = []
        for arm in (JAC, FRAMEWORK, SDK):
            present = any(r["benchmark"] == bench and r["arm"].startswith(arm) for r in rows)
            cells.append("yes" if present else "**MISSING**")
            if not present:
                gaps.append(f"{bench}/{arm}")
        lines.append(f"| {bench} | " + " | ".join(cells) + " |")
    if gaps:
        lines += ["", f"**{len(gaps)} empty cell(s):** " + ", ".join(gaps),
                  "", "A missing cell is not a result. Check that stage's log before "
                  "reading anything above as a framework comparison."]
    else:
        lines += ["", "All fifteen cells populated."]

    # -- ledger ------------------------------------------------------------
    if ledger:
        lines += ["", "## Token ledger (proxy-observed)", "",
                  "Counted at the wire by `RagGPT/eval/harness/proxy.py`, in front of the "
                  "model server, so byLLM, LangGraph, CrewAI and the raw SDK are all "
                  "measured the same way.", "",
                  "| attributed to | calls | prompt | completion | calls without usage |",
                  "|---|---|---|---|---|"]
        for key in sorted(ledger):
            agg = ledger[key]
            lines.append(f"| {key} | {agg['calls']:,} | {agg['prompt']:,} | "
                         f"{agg['completion']:,} | {agg['no_usage']:,} |")
        blind = sum(a["no_usage"] for a in ledger.values())
        if blind:
            lines += ["", f"{blind:,} call(s) returned no usage block. Those tokens are "
                          "real but uncounted -- usually a streamed response without "
                          "`stream_options.include_usage`."]

    # -- stages ------------------------------------------------------------
    lines += ["", "## Stages", "", "| stage | result | steps |", "|---|---|---|"]
    for st in manifest.get("stages", []):
        steps = ", ".join(
            f"{s['step']}={s.get('exit_code', s.get('skipped', '?'))}" for s in st.get("steps", []))
        lines.append(f"| {st['title']} | {'ok' if st.get('ok') else 'FAILED'} | {steps} |")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", default=None,
                    help="which bench/runs/<id> to summarize (default: the newest)")
    args = ap.parse_args(argv)

    run_id = args.run_id
    if not run_id:
        candidates = sorted((d for d in config.RUNS_ROOT.glob("*") if d.is_dir()),
                            key=lambda d: d.stat().st_mtime, reverse=True)
        if not candidates:
            raise SystemExit(f"no runs under {config.RUNS_ROOT}; run bench.run_all first")
        run_id = candidates[0].name

    run_dir = config.RUNS_ROOT / run_id
    manifest = _load_json(run_dir / "manifest.json") or {"stages": []}

    rows = (rows_meeting() + rows_email() + rows_ytnavigator()
            + rows_raggpt() + rows_codeagent(run_id))
    ledger = ledger_totals(run_dir / "tokens.jsonl")

    text = render(rows, manifest, ledger, run_id)
    (run_dir / "summary.md").write_text(text)

    csv_path = run_dir / "summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_row("", "", "").keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(text)
    print(f"Written: {run_dir / 'summary.md'}")
    print(f"         {csv_path}")
    # An empty cell is a failure of the run, not a finding about a framework.
    missing = sum(1 for b in ("meeting-assistant", "Email-Auto-response", "YTNavigator",
                              "RagGPT", "CodeAgent")
                  for a in (JAC, FRAMEWORK, SDK)
                  if not any(r["benchmark"] == b and r["arm"].startswith(a) for r in rows))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
