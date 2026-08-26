#!/usr/bin/env python3
"""Choose an *easy* instance set, from what the leaderboard already knows.

    python3 select_easy.py --count 8

`select_instances.py` draws the opposite kind of set: instances where our own
implementations disagreed, which by construction are the ones at least one of
them already failed. That is the right set for telling two agents apart and the
wrong one for a first end-to-end run -- a set where everything fails tells you
nothing about the bridge, the container runtime or the model wiring.

So this script uses a prior nobody here had to generate: **every SWE-bench Lite
submission on the official leaderboard**, at
`github.com/SWE-bench/experiments/evaluation/lite/*/results/results.json`, each
of which lists exactly which of the 300 instances that system resolved. Counting
across submissions gives a per-instance resolve rate -- an empirical difficulty
that no single agent's opinion decides. As of the cached snapshot, 84 systems
have submitted, and the spread is real: the top instances are resolved by ~94% of
them and 60 instances by fewer than 10%.

Two rates are kept, and they answer different questions:

  * `rate_recent` -- over submissions from `--since` (default 2025) onward. These
    are systems built on models of roughly the class we run, so this is the one
    that predicts our runs, and it is what the draw ranks on.
  * `rate_all` -- over every submission since 2023, including the original RAG
    baselines that resolved 1-4% of Lite. Kept as a tie-break and printed,
    because an instance that even a 2023 retrieval baseline got is easy in a way
    that is not about model strength.

What this set is *not*: representative. It is the top of a difficulty ranking, so
a resolve rate measured on it is an upper bound and is not comparable to a rate
over the full 300 -- the same warning `select_instances.py` carries, from the
other end of the same distribution. It is for shaking out the pipeline and for a
cheap smoke comparison between implementations, not for a score anyone quotes.

The counts are cached in `leaderboard_lite.json` so the draw is reproducible
offline and reviewable in the diff; `--refresh` re-fetches them from GitHub.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "leaderboard_lite.json"

API = ("https://api.github.com/repos/SWE-bench/experiments"
       "/contents/evaluation/lite")
RAW = ("https://raw.githubusercontent.com/SWE-bench/experiments/main"
       "/evaluation/lite/{}/results/results.json")


def repo_of(instance_id: str) -> str:
    return instance_id.split("__")[0]


def fetch_json(url: str, timeout: int = 60):
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        return json.load(fh)


def fetch_leaderboard() -> dict:
    """{submission -> [resolved instance ids]} for every Lite submission."""
    names = [e["name"] for e in fetch_json(API) if e["type"] == "dir"]

    def one(name: str):
        try:
            return name, fetch_json(RAW.format(name)).get("resolved", [])
        except Exception as e:  # noqa: BLE001 - a submission without results
            print(f"  skipped {name}: {e}", file=sys.stderr)
            return name, None

    with ThreadPoolExecutor(8) as ex:
        pairs = list(ex.map(one, names))
    return {n: r for n, r in pairs if r is not None}


def rates(runs: dict, since: str) -> dict[str, dict]:
    """Per-instance resolve rate, over everything and over the recent window."""
    recent = {n: r for n, r in runs.items() if n[:4] >= since}
    all_c, rec_c = Counter(), Counter()
    for res in runs.values():
        all_c.update(res)
    for res in recent.values():
        rec_c.update(res)
    n_all, n_rec = len(runs), max(1, len(recent))
    return {
        iid: {
            "rate_all": all_c[iid] / n_all,
            "rate_recent": rec_c[iid] / n_rec,
            "n_all": n_all,
            "n_recent": len(recent),
        }
        for iid in all_c
    }


def draw(scored: dict[str, dict], count: int, keep: list[str],
         max_per_repo: int, min_rate: float) -> list[str]:
    """The easiest `count`, ranked on the recent window, capped per repo.

    The cap is what stops the set from being all django: django is 114 of Lite's
    300 and holds most of the top of the ranking, and eight django instances
    would measure one project's conventions rather than the agent.
    """
    order = sorted(
        (i for i, s in scored.items() if s["rate_recent"] >= min_rate),
        key=lambda i: (-scored[i]["rate_recent"], -scored[i]["rate_all"], i),
    )
    used: Counter = Counter()
    chosen: list[str] = []
    for iid in keep:                       # pins ignore the cap, by request
        if iid in scored and iid not in chosen:
            chosen.append(iid)
            used[repo_of(iid)] += 1
    for iid in order:
        if len(chosen) >= count:
            break
        if iid in chosen or used[repo_of(iid)] >= max_per_repo:
            continue
        chosen.append(iid)
        used[repo_of(iid)] += 1
    if len(chosen) < count:
        # Rather than silently return fewer: relax the cap, and say so in main().
        for iid in order:
            if len(chosen) >= count:
                break
            if iid not in chosen:
                chosen.append(iid)
    return chosen[:count]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Pick the instances the leaderboard says are easiest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--out", type=Path, default=HERE / "instances.txt")
    p.add_argument("--keep", nargs="*", default=[],
                   help="instance ids that must appear, cap or not")
    p.add_argument("--max-per-repo", type=int, default=2)
    p.add_argument("--min-rate", type=float, default=0.6,
                   help="floor on the recent-window resolve rate")
    p.add_argument("--since", default="2025",
                   help="submissions from this year on are the recent window")
    p.add_argument("--refresh", action="store_true",
                   help="re-fetch the leaderboard instead of using the cache")
    p.add_argument("--check", action="store_true",
                   help="verify --out matches the rules; write nothing")
    args = p.parse_args(argv)

    if args.refresh or not CACHE.exists():
        print("fetching SWE-bench/experiments ...", file=sys.stderr)
        runs = fetch_leaderboard()
        # One line per submission: 84 lines instead of 17k, so the snapshot
        # this draw rests on is reviewable in a diff.
        body = ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}"
                          for k, v in sorted(runs.items()))
        CACHE.write_text("{\n" + body + "\n}\n", encoding="utf-8")
        print(f"  cached {len(runs)} submissions -> {CACHE}", file=sys.stderr)
    else:
        runs = json.loads(CACHE.read_text(encoding="utf-8"))

    scored = rates(runs, args.since)
    chosen = draw(scored, args.count, args.keep, args.max_per_repo,
                  args.min_rate)
    body = "".join(f"{i}\n" for i in sorted(chosen))

    if args.check:
        if not args.out.exists():
            print(f"{args.out} does not exist", file=sys.stderr)
            return 1
        if args.out.read_text(encoding="utf-8") != body:
            print(f"{args.out} is stale; re-run without --check",
                  file=sys.stderr)
            return 1
        print(f"{args.out} matches the rules ({len(chosen)} instances)")
        return 0

    args.out.write_text(body, encoding="utf-8")
    n_rec = next(iter(scored.values()))["n_recent"]
    print(f"{len(runs)} leaderboard submissions ({n_rec} since {args.since}), "
          f"{len(scored)} instances resolved by at least one "
          f"-> {len(chosen)} chosen")
    print(f"  {args.out}")
    print(f"  {'instance':38s} {'recent':>7s} {'all-time':>9s}")
    for iid in sorted(chosen, key=lambda i: -scored[i]["rate_recent"]):
        s = scored[iid]
        print(f"  {iid:38s} {s['rate_recent']*100:6.0f}% {s['rate_all']*100:8.0f}%")
    by_repo = Counter(repo_of(i) for i in chosen)
    print("  repos : " + ", ".join(f"{k} {v}" for k, v in sorted(by_repo.items())))
    over = {k: v for k, v in by_repo.items() if v > args.max_per_repo}
    for repo, n in sorted(over.items()):
        print(f"  note  : {repo} is {n}/{len(chosen)}, over the "
              f"{args.max_per_repo} cap -- the rate floor left nothing else")
    print("  warning: the top of a difficulty ranking is not a sample of the "
          "benchmark; a rate measured here is an upper bound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
