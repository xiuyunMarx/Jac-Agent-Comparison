"""Delete every result artifact; keep every input.

The distinction is not "is it in an output directory" -- several inputs live in
one. `Email-Auto-response/eval/out/benchmark_slide.pptx` is hand-made and
nothing regenerates it. `RagGPT/eval/dataset/dataset.jsonl` was synthesized with
gpt-4.1 and cannot be rebuilt offline, and rebuilding it would change the
benchmark rather than re-run it. `meeting-assistant/byLLM/meeting_notes.txt`
looks like leftover output and is actually the transcript that arm reads when
run standalone. Each of those is listed below with the reason, because the cost
of getting one wrong is discovering it after the deletion.

Run with --dry-run first; it prints exactly what would go.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from . import config

R = config.REPO_ROOT

#: (path-or-glob, why). Directories are removed whole.
RESULTS: list[tuple[str, str]] = [
    # -- Email-Auto-response ------------------------------------------------
    ("Email-Auto-response/byLLM/mock_output", "agent run output"),
    ("Email-Auto-response/CrewAI-LangGraph/mock_output", "agent run output"),
    ("Email-Auto-response/openai_sdk/mock_output", "agent run output"),
    ("Email-Auto-response/eval/out/scores_*.json", "scorer output"),
    ("Email-Auto-response/eval/out/summary.json", "scorer output"),
    ("Email-Auto-response/runs", "hand-archived snapshot of a prior gpt-4o-mini run"),
    # -- meeting-assistant --------------------------------------------------
    ("meeting-assistant/eval/runs", "per-run results and work dirs"),
    ("meeting-assistant/eval/out", "scorer output"),
    # -- YTNavigator --------------------------------------------------------
    ("YTNavigator/eval/out", "agent results + report"),
    ("YTNavigator/YT-Navigator/logs/*.jsonl", "Django request log noise"),
    ("YTNavigator/eval/pgdata", "Postgres data dir: not portable, e2e.py rebuilds it"),
    ("YTNavigator/eval/pgdata.log", "postmaster log"),
    # -- RagGPT -------------------------------------------------------------
    ("RagGPT/eval/results", "raw runs, proxy log, judge cache, server logs, report"),
    # -- CodeAgent ----------------------------------------------------------
    ("CodeAgent/swebench_bridge/results", "predictions, runs, eval logs, harness reports"),
    ("CodeAgent/swebench_bridge/*.log", "hand-captured console transcripts"),
    ("CodeAgent/case_study/lite-01", "generated study"),
    ("CodeAgent/case_study/three-way", "generated study"),
    ("CodeAgent/byLLM/llm_calls", "per-call LLM transcripts"),
    ("CodeAgent/byLLM/history_records", "hand-archived transcripts from earlier runs"),
]

#: Files that sit inside a deleted directory but must survive it.
RESCUE: list[tuple[str, str]] = [
    ("Email-Auto-response/eval/out/benchmark_slide.pptx",
     "hand-made; no code regenerates it"),
]

#: Never touched. Listed so the reasoning is on the record, not just the effect.
PRESERVED: list[tuple[str, str]] = [
    ("Email-Auto-response/mock_mailbox/datasets", "the input mailboxes"),
    ("meeting-assistant/datasets", "labeled input transcripts"),
    ("meeting-assistant/byLLM/meeting_notes.txt", "the arm's standalone input transcript"),
    ("meeting-assistant/CrewAI/meeting_notes.txt", "the arm's standalone input transcript"),
    ("YTNavigator/datasets", "synthetic channel + ground-truth questions"),
    ("RagGPT/eval/dataset/dataset.jsonl", "gpt-4.1-synthesized; not rebuildable offline"),
    ("RagGPT/*/faiss_index", "committed or deterministic derived index; rebuilding adds a variable"),
    ("CodeAgent/case_study/instances.txt", "the pinned instance set the comparison rests on"),
    ("CodeAgent/SWE-bench", "the vendored grading harness"),
]

CACHE_DIR_NAMES = ("__pycache__", ".pytest_cache")

#: Output directory names the re-run will recreate. Adding them to .gitignore
#: keeps ~110 MB of results from being recommitted; most of what we just
#: deleted was tracked.
GITIGNORE_LINES = [
    "",
    "# Benchmark output (bench/clean.py removes these; do not commit them)",
    "**/mock_output/",
    "**/eval/out/",
    "**/eval/runs/",
    "**/eval/results/",
    "**/swebench_bridge/results/",
    "bench/runs/",
    "__pycache__/",
    ".pytest_cache/",
]


def _expand(pattern: str) -> list[Path]:
    if any(ch in pattern for ch in "*?["):
        return sorted(R.glob(pattern))
    p = R / pattern
    return [p] if p.exists() else []


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human(n: float) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}G"


def stop_postgres(dry_run: bool) -> None:
    """A live postmaster on eval/pgdata must be stopped before the dir goes.

    e2e.py starts it from whichever conda env has pgvector-capable binaries, so
    look for the running process rather than assuming a pg_ctl on PATH.
    """
    pgdata = R / "YTNavigator" / "eval" / "pgdata"
    if not pgdata.exists():
        return
    try:
        out = subprocess.run(["pgrep", "-af", f"postgres.*{pgdata}"],
                             capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        out = ""
    if not out:
        return
    binary = out.splitlines()[0].split()[1]
    pg_ctl = Path(binary).parent / "pg_ctl"
    print(f"  postgres is live on {pgdata}")
    if dry_run:
        print(f"  would run: {pg_ctl} -D {pgdata} stop")
        return
    if pg_ctl.exists():
        print(f"  stopping: {pg_ctl} -D {pgdata} stop")
        subprocess.run([str(pg_ctl), "-D", str(pgdata), "stop", "-m", "fast"],
                       capture_output=True)
    else:
        raise SystemExit(
            f"a postmaster is running on {pgdata} but {pg_ctl} does not exist.\n"
            "Stop it yourself and rerun, or the data directory cannot be removed safely.")


def clean_caches(dry_run: bool) -> int:
    freed = 0
    for name in CACHE_DIR_NAMES:
        for path in R.rglob(name):
            if ".venv" in path.parts or "SWE-bench" in path.parts:
                continue
            freed += _size(path)
            if not dry_run:
                shutil.rmtree(path, ignore_errors=True)
    return freed


def update_gitignore(dry_run: bool) -> None:
    gitignore = R / ".gitignore"
    existing = gitignore.read_text() if gitignore.exists() else ""
    missing = [ln for ln in GITIGNORE_LINES if ln and ln not in existing]
    if not missing:
        return
    print(f"\n.gitignore: adding {len(missing)} pattern(s) so results are not recommitted")
    if not dry_run:
        gitignore.write_text(existing.rstrip("\n") + "\n" + "\n".join(GITIGNORE_LINES) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="print the manifest, delete nothing")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--keep-caches", action="store_true", help="leave __pycache__/.pytest_cache")
    args = ap.parse_args(argv)

    targets: list[tuple[Path, str, int]] = []
    for pattern, why in RESULTS:
        for path in _expand(pattern):
            targets.append((path, why, _size(path)))

    if not targets:
        print("Nothing to clean: no result artifacts found.")
    else:
        total = sum(t[2] for t in targets)
        print(f"{'Would delete' if args.dry_run else 'Deleting'} {len(targets)} path(s), "
              f"{_human(total)}:\n")
        for path, why, size in targets:
            print(f"  {_human(size):>8}  {path.relative_to(R)}")
            print(f"            {why}")

    print("\nPreserved (inputs, despite where they live):")
    for pattern, why in PRESERVED:
        print(f"  {pattern}\n            {why}")

    rescued = [(R / p, why) for p, why in RESCUE if (R / p).exists()]
    if rescued:
        print("\nRescued from a deleted directory:")
        for path, why in rescued:
            print(f"  {path.relative_to(R)}\n            {why}")

    if args.dry_run:
        stop_postgres(dry_run=True)
        clean_caches(dry_run=True)
        update_gitignore(dry_run=True)
        print("\n(dry run -- nothing was deleted)")
        return 0

    if not args.yes:
        reply = input("\nProceed? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted; nothing deleted.")
            return 1

    print()
    stop_postgres(dry_run=False)

    stash = {}
    for path, _ in rescued:
        stash[path] = path.read_bytes()

    for path, _, _ in targets:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)

    for path, _ in rescued:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stash[path])
        print(f"  restored {path.relative_to(R)}")

    if not args.keep_caches:
        freed = clean_caches(dry_run=False)
        print(f"  removed python caches ({_human(freed)})")

    update_gitignore(dry_run=False)
    print("\nClean. Inputs are intact; every result is gone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
