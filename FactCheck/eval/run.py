"""Drive the FactCheck arms over hover_claims.tsv, one result JSON per (arm, claim).

    python eval/run.py --arms jac openai langgraph nooa [--limit N] [--workers 2] [--out eval/out]

Each claim is one subprocess of the arm's entry point with the same knobs the
arm reads on its own (FC_*), the shared wiki_cache, and a per-call token log:
FC_TRACE for the Python arms, BYLLM_USAGE_LOG for the Jac arm (jac-main build).
The result records the printed verdict, rounds, findings, call count, tokens
and wall time. Existing result files are skipped unless --force.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # FactCheck/
PY = os.environ.get("FC_PY", str(Path.home() / "miniconda3/envs/jaseci/bin/python"))
JAC = os.environ.get("FC_JAC", str(Path.home() / "miniconda3/envs/jac-main/bin/jac"))
PY_NOOA = os.environ.get("FC_PY_NOOA", str(Path.home() / "miniconda3/envs/nooa/bin/python"))

ARMS = {
    "jac": {"cwd": ROOT / "Jac", "cmd": [JAC, "run", "fact_check.jac"], "log_env": "BYLLM_USAGE_LOG"},
    "openai": {"cwd": ROOT, "cmd": [PY, "OpenaiSDK/fact_check.py"], "log_env": "FC_TRACE"},
    "langgraph": {"cwd": ROOT, "cmd": [PY, "LangGraph/fact_check.py"], "log_env": "FC_TRACE"},
    "nooa": {"cwd": ROOT, "cmd": [PY_NOOA, "NOOA/fact_check.py"], "log_env": "FC_TRACE"},
}


def load_claims(path: Path, limit: int | None) -> list[dict]:
    claims = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        label, hops, claim = line.split("\t", 2)
        claims.append({"idx": len(claims) + 1, "gold": label.strip(), "hops": int(hops), "claim": claim.strip()})
    return claims[:limit] if limit else claims


def read_log(path: Path) -> dict:
    """Calls and tokens from either log format (both are one JSON object per line)."""
    calls = prompt = completion = 0
    retries = 0
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            calls += 1
            usage = row.get("usage") or row     # Python arms nest usage; byLLM flattens it
            prompt += int(usage.get("prompt_tokens") or 0)
            completion += int(usage.get("completion_tokens") or 0)
            if str(row.get("stage", "")).endswith("/retry"):
                retries += 1
    return {"calls": calls, "prompt_tokens": prompt, "completion_tokens": completion, "retries": retries}


def run_one(arm: str, item: dict, out: Path, timeout: int) -> dict:
    spec = ARMS[arm]
    tag = f"{arm}_{item['idx']:03d}"
    log = out / f"{tag}.calls.jsonl"
    log.unlink(missing_ok=True)
    env = dict(os.environ)
    env[spec["log_env"]] = str(log)
    env.setdefault("FC_CACHE_DIR", str(ROOT / "wiki_cache"))
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            spec["cmd"] + [item["claim"]], cwd=spec["cwd"], env=env,
            capture_output=True, text=True, timeout=timeout,
        )
        exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        exit_code, stdout, stderr = -9, (e.stdout or ""), f"timeout after {timeout}s\n" + (e.stderr or "")
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
    wall = time.monotonic() - t0
    (out / f"{tag}.stdout.txt").write_text(stdout + ("\n--- stderr ---\n" + stderr if stderr else ""))

    verdict = re.search(r"^Verdict: (\w+)", stdout, re.M)
    rounds = re.search(r"^Rounds: (\d+), findings: (\d+)", stdout, re.M)
    result = {
        "arm": arm, **item,
        "verdict": verdict.group(1) if verdict else None,
        "rounds": int(rounds.group(1)) if rounds else None,
        "findings": int(rounds.group(2)) if rounds else None,
        "exit": exit_code,
        "error": None if exit_code == 0 and verdict else (stderr.strip().splitlines() or ["no verdict printed"])[-1][:300],
        "wall_s": round(wall, 1),
        **read_log(log),
    }
    result["correct"] = result["verdict"] == item["gold"]
    (out / f"{tag}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--claims", default=str(ROOT / "hover_claims.tsv"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2, help="concurrent claims per arm")
    ap.add_argument("--out", default=str(ROOT / "eval/out"))
    ap.add_argument("--timeout", type=int, default=1500, help="seconds per claim")
    ap.add_argument("--serial", action="store_true", help="one process at a time overall (arms in order); overrides --workers")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("OLLAMA_API_KEY"):
        sys.exit("OLLAMA_API_KEY is not set (set -a; . ../.env; set +a)")
    # Absolute: the Jac arm runs with Jac/ as its cwd and byLLM would silently
    # drop a usage log whose relative path does not resolve from there.
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    claims = load_claims(Path(args.claims), args.limit)

    jobs = []
    for arm in args.arms:
        for item in claims:
            prior = out / f"{arm}_{item['idx']:03d}.json"
            # A successful result is kept; a failed one (crash, kill, timeout) is retried.
            if not args.force and prior.exists() and json.loads(prior.read_text()).get("error") is None:
                continue
            jobs.append((arm, item))
    print(f"{len(jobs)} runs ({len(claims)} claims x {args.arms}), {args.workers} workers per arm", flush=True)

    if args.serial:
        # One process at a time overall: arm after arm, claim after claim.
        pools = {arm: ThreadPoolExecutor(max_workers=1) for arm in args.arms[:1]}
        shared = next(iter(pools.values()))
        futures = {shared.submit(run_one, arm, item, out, args.timeout): (arm, item) for arm, item in jobs}
    else:
        # One pool per arm so every arm progresses at the same rate against the endpoint.
        pools = {arm: ThreadPoolExecutor(max_workers=args.workers) for arm in args.arms}
        futures = {pools[arm].submit(run_one, arm, item, out, args.timeout): (arm, item) for arm, item in jobs}
    done = 0
    for fut in as_completed(futures):
        arm, item = futures[fut]
        done += 1
        try:
            r = fut.result()
            status = "ok " if r["correct"] else ("ERR" if r["error"] else "   ")
            print(f"[{done}/{len(jobs)}] {arm:9} #{item['idx']:03d} {status} gold={item['gold']:13} got={r['verdict']} "
                  f"calls={r['calls']} tok={r['prompt_tokens']}+{r['completion_tokens']} {r['wall_s']}s"
                  + (f"  error: {r['error']}" if r["error"] else ""), flush=True)
        except Exception as e:  # a harness bug, not an arm failure
            print(f"[{done}/{len(jobs)}] {arm:9} #{item['idx']:03d} HARNESS ERROR {e!r}", flush=True)
    for pool in pools.values():
        pool.shutdown()


if __name__ == "__main__":
    main()
