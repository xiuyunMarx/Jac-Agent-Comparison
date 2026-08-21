"""Eval runner: drives every (system x repeat x item) turn through the proxy.

Sequential by design — the proxy attributes LLM calls to whatever marker was
set last, so only one turn may be in flight at a time.

Resume unit is (system, item, repeat): an item is done when ALL its turns are
recorded; partially recorded items are redone whole under a fresh session id
(scoring keeps only the last occurrence per turn).

Run:  /home/xiaoyu/miniconda3/envs/jaseci/bin/python run_eval.py \
        [--systems langgraph,jac,jac-byllm-router] [--repeats 3] [--limit N] \
        [--categories rag_qa,...]
"""

import argparse
import atexit
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402
import drivers  # noqa: E402

PROXY_HEALTH = f"http://127.0.0.1:{common.PROXY_PORT}/__health"
PROXY_MARK = f"http://127.0.0.1:{common.PROXY_PORT}/__mark"
LOCK_PATH = common.RESULTS_DIR / ".run_eval.lock"


def acquire_lock() -> None:
    """Refuse to run concurrently — two runners corrupt proxy attribution."""
    if LOCK_PATH.exists():
        try:
            pid = int(LOCK_PATH.read_text().strip())
            os.kill(pid, 0)  # raises if not running
            raise SystemExit(
                f"another run_eval (pid {pid}) is already running — "
                f"wait for it or kill it, then delete {LOCK_PATH}")
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # stale lock
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(str(os.getpid()))
    atexit.register(lambda: LOCK_PATH.unlink(missing_ok=True))


def _kill_stray_proxies() -> None:
    """Kill any of our own leftover proxy.py processes (e.g. from tests)."""
    me = os.getpid()
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit() or int(pid_dir.name) == me:
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if "proxy.py" in cmdline and "python" in cmdline:
            try:
                os.kill(int(pid_dir.name), 15)
                print(f"killed stray proxy pid {pid_dir.name}")
            except (ProcessLookupError, PermissionError):
                pass
    time.sleep(1)


def ensure_proxy() -> subprocess.Popen | None:
    try:
        health = requests.get(PROXY_HEALTH, timeout=2).json()
        # A proxy is only trustworthy if it forwards where this run expects and
        # logs where scoring will look — anything else is a stray (test)
        # instance. The upstream is compared against the configured one, not a
        # literal, so a shared proxy in front of a local model is reused rather
        # than killed and replaced with one pointed at OpenAI.
        if (health.get("upstream") == common.PROXY_UPSTREAM
                and health.get("log") == str(common.PROXY_LOG_PATH)):
            print("proxy already running (verified)")
            return None
        print(f"WRONG proxy on port {common.PROXY_PORT} "
              f"(upstream={health.get('upstream')}); replacing it")
        _kill_stray_proxies()
    except requests.RequestException:
        pass
    env = dict(os.environ)
    env["PROXY_LOG"] = str(common.PROXY_LOG_PATH)
    env["PROXY_PORT"] = str(common.PROXY_PORT)
    env["PROXY_UPSTREAM"] = common.PROXY_UPSTREAM
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "proxy.py")],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    for _ in range(25):
        try:
            if requests.get(PROXY_HEALTH, timeout=2).ok:
                print("proxy started")
                return proc
        except requests.RequestException:
            time.sleep(0.4)
    raise SystemExit("could not start proxy")


def mark(system: str, item_id: str, repeat: int, turn: int, attempt: str) -> None:
    requests.post(PROXY_MARK, timeout=10, json={
        "system": system, "item_id": item_id, "repeat": repeat, "turn": turn,
        "attempt": attempt})


def load_done() -> dict[tuple, set[int]]:
    """(system, item_id, repeat) -> turn indices that succeeded (errors retry)."""
    done: dict[tuple, set[int]] = {}
    for row in common.read_jsonl(common.RAW_RUNS_PATH):
        if not row.get("error"):
            done.setdefault((row["system"], row["item_id"], row["repeat"]),
                            set()).add(row["turn"])
    return done


def run_item(driver, system: str, item: dict, repeat: int) -> None:
    session_id = f"{item['id']}.r{repeat}.{uuid.uuid4().hex[:8]}"
    for turn_idx, turn in enumerate(item["turns"]):
        attempt = uuid.uuid4().hex[:12]
        mark(system, item["id"], repeat, turn_idx, attempt)
        t0 = time.time()
        try:
            result = driver.interact(turn["message"], session_id)
            error = ""
        except Exception as e:  # recorded, not fatal to the run
            result = {"agent": "", "response": ""}
            error = f"{type(e).__name__}: {e}"[:500]
        latency_ms = int((time.time() - t0) * 1000)
        common.append_jsonl(common.RAW_RUNS_PATH, {
            "ts": time.time(),
            "system": system,
            "item_id": item["id"],
            "category": item["category"],
            "repeat": repeat,
            "turn": turn_idx,
            "gold_agent": turn["gold_agent"],
            "routed_agent": result["agent"],
            "response": result["response"],
            "latency_ms": latency_ms,
            "session_id": session_id,
            "attempt": attempt,
            "error": error,
        })
        status = result["agent"] or ("ERROR " + error[:60])
        print(f"  {system} r{repeat} {item['id']} t{turn_idx}: {status} ({latency_ms}ms)",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=",".join(common.SYSTEMS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="cap items per category (smoke)")
    ap.add_argument("--categories", default=",".join(common.CATEGORIES))
    args = ap.parse_args()

    common.require_api_key()
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    for s in systems:
        if s not in common.SYSTEMS:
            raise SystemExit(f"unknown system {s!r}; known: {list(common.SYSTEMS)}")
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    items = [it for it in common.read_jsonl(common.DATASET_PATH)
             if it["category"] in categories]
    if args.limit:
        capped: dict[str, int] = {}
        limited = []
        for it in items:
            if capped.get(it["category"], 0) < args.limit:
                capped[it["category"]] = capped.get(it["category"], 0) + 1
                limited.append(it)
        items = limited
    if not items:
        raise SystemExit(f"no dataset items — run synthesize.py first ({common.DATASET_PATH})")
    print(f"{len(items)} items x {len(systems)} systems x {args.repeats} repeats")

    acquire_lock()
    # The proxy stays up after the run (identity-verified on every start), so a
    # finishing run can never yank it out from under anything else.
    ensure_proxy()
    done = load_done()
    try:
        for system in systems:
            pending = [(it, rep) for rep in range(1, args.repeats + 1) for it in items
                       if set(range(len(it["turns"]))) - done.get((system, it["id"], rep), set())]
            if not pending:
                print(f"== {system}: all done, skipping")
                continue
            print(f"== {system}: {len(pending)} item-runs")
            driver = drivers.make_driver(system, common.RESULTS_DIR)
            driver.start()
            try:
                for it, rep in pending:
                    run_item(driver, system, it, rep)
            finally:
                driver.stop()
    finally:
        LOCK_PATH.unlink(missing_ok=True)
    print("run complete ->", common.RAW_RUNS_PATH)


if __name__ == "__main__":
    main()
