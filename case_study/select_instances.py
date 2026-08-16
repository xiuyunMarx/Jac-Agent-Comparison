#!/usr/bin/env python3
"""Choose the instances a comparison run should use.

    python3 select_instances.py --from lite-01/divergence.csv --count 30

Draws from a `divergence.csv` -- the instances where the implementations already
disagreed -- because an instance every side resolved, or none did, says nothing
about the difference between them. That makes the result a deliberately *hard,
discriminating* set rather than a sample of the benchmark: a resolve rate
measured on it is not comparable to a rate over the full split, and any study
built from it has to say so.

Two things it will not draw:

  * **Split verdicts on byte-identical patches.** The same diff graded both ways
    is a flake in the harness, not a difference between the agents.
  * **Instances a side lost to infrastructure.** A container that never came up
    or a provider 400 tells you about this machine and that API, not about the
    implementation. A *timeout* is not in this category and is kept: the agent
    spent its own budget and came back with nothing, which is a real result.

Then it balances. The draw is even across the winning sides, because an
unbalanced set lets a newly-added implementation score well merely by resembling
whichever side is over-represented. Repos are capped at their share of the pool
for the same reason -- django is 38% of SWE-bench Lite and around half the
divergence, and left alone it would decide the whole comparison.

`--keep` pins instances that must appear whatever the balancing says, for when a
study wants continuity with an earlier one.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Run-error kinds that make an instance evidence about the machine rather than
# about the agent. `timeout` is deliberately absent -- see the module docstring.
INFRA_ERRORS = {"infra(container)", "infra(pull)", "api-400", "api-rate-limit"}


def repo_of(instance_id: str) -> str:
    return instance_id.split("__")[0]


def frameworks_in(header: list[str]) -> list[str]:
    return [c[:-len("_status")] for c in header if c.endswith("_status")]


def excluded(row: dict, names: list[str]) -> str:
    """Why this instance is not evidence, or "" if it is."""
    if row.get("identical_patches") == "True":
        return "byte-identical patches, split verdict"
    for fw in names:
        kind = row.get(f"{fw}_run_error", "")
        if kind in INFRA_ERRORS:
            return f"{fw} lost to infrastructure ({kind})"
    return ""


def draw(pool: list[dict], count: int, keep: set[str],
         max_repo_fraction: float = 0.5) -> list[str]:
    """Balance across winning sides, then across repos, up to `count`."""
    by_side: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        by_side[row["resolved_by"]].append(row)
    sides = sorted(by_side)
    if not sides:
        return []

    # Even split, remainder to the earlier sides so the total is exactly `count`.
    base, extra = divmod(count, len(sides))
    quota = {s: base + (1 if i < extra else 0) for i, s in enumerate(sides)}

    # A flat ceiling, not each repo's share of the pool. Tracking the pool would
    # not bind at all here -- django is over half of it -- and a comparison
    # decided by one project's conventions is a comparison of that project.
    # Whether the ceiling can actually be honoured depends on the pool; `main`
    # reports the achieved shares either way.
    ceiling = max(1, int(count * max_repo_fraction))
    cap = {repo: ceiling for repo in {repo_of(r["instance_id"]) for r in pool}}

    chosen: list[str] = []
    used: Counter = Counter()
    for side in sides:
        rows = sorted(by_side[side], key=lambda r: r["instance_id"])
        pinned = [r for r in rows if r["instance_id"] in keep]
        rest = [r for r in rows if r["instance_id"] not in keep]
        # Least-represented repo first, so the cap binds on the crowded ones.
        rest.sort(key=lambda r: (used[repo_of(r["instance_id"])], r["instance_id"]))
        picked: list[dict] = []
        for row in pinned + rest:
            if len(picked) >= quota[side]:
                break
            repo = repo_of(row["instance_id"])
            if row["instance_id"] not in keep and used[repo] >= cap.get(repo, count):
                continue
            picked.append(row)
            used[repo] += 1
        # A side that cannot fill its quota under the cap gets the rest anyway:
        # a smaller balanced set beats a set that silently drops a whole side.
        if len(picked) < quota[side]:
            for row in pinned + rest:
                if len(picked) >= quota[side]:
                    break
                if row not in picked:
                    picked.append(row)
                    used[repo_of(row["instance_id"])] += 1
        chosen += [r["instance_id"] for r in picked]

    # Redistribute any shortfall. An even quota cannot be met by an uneven pool
    # -- 36 wanted from a 17/19 split gives each side 18, and the byLLM side can
    # only supply 17 -- so without this, asking for the entire pool returns one
    # instance fewer than the pool, which is the one case the caller is most
    # entitled to expect exactly.
    if len(chosen) < count:
        taken = set(chosen)
        spare = sorted((r for r in pool if r["instance_id"] not in taken),
                       key=lambda r: (used[repo_of(r["instance_id"])],
                                      r["instance_id"]))
        for row in spare:
            if len(chosen) >= count:
                break
            chosen.append(row["instance_id"])
            used[repo_of(row["instance_id"])] += 1
    return sorted(chosen)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Pick a balanced, discriminating instance set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--from", dest="source", type=Path,
                   default=HERE / "lite-01" / "divergence.csv",
                   help="a divergence.csv written by build_study.py")
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--keep", nargs="*", default=[],
                   help="instance ids that must be in the draw")
    p.add_argument("--max-repo-fraction", type=float, default=0.5,
                   help="ceiling on any one repo's share of the draw; the pool "
                        "can force it higher, and the summary says when it did")
    p.add_argument("--out", type=Path, default=HERE / "instances.txt")
    p.add_argument("--check", action="store_true",
                   help="verify --out matches the rules; write nothing")
    args = p.parse_args(argv)

    if not args.source.exists():
        raise SystemExit(f"no divergence file at {args.source}; "
                         "build a study first with build_study.py")
    with args.source.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        names = frameworks_in(reader.fieldnames or [])

    pool, dropped = [], []
    for row in rows:
        why = excluded(row, names)
        (dropped if why else pool).append((row, why))
    pool = [r for r, _ in pool]

    chosen = draw(pool, args.count, set(args.keep), args.max_repo_fraction)
    body = "\n".join(chosen) + "\n"

    if args.check:
        if not args.out.exists():
            print(f"{args.out} does not exist", file=sys.stderr)
            return 1
        if args.out.read_text(encoding="utf-8") != body:
            print(f"{args.out} is stale; re-run without --check", file=sys.stderr)
            return 1
        print(f"{args.out} matches the rules ({len(chosen)} instances)")
        return 0

    args.out.write_text(body, encoding="utf-8")
    by_repo = Counter(repo_of(i) for i in chosen)
    by_side = Counter(r["resolved_by"] for r in pool if r["instance_id"] in chosen)
    print(f"{len(rows)} diverging, {len(dropped)} not evidence, "
          f"{len(pool)} in the pool -> {len(chosen)} chosen")
    print(f"  {args.out}")
    print("  sides : " + ", ".join(f"{k} {v}" for k, v in sorted(by_side.items())))
    print("  repos : " + ", ".join(
        f"{k} {v}" for k, v in sorted(by_repo.items(), key=lambda kv: (-kv[1], kv[0]))))
    ceiling = max(1, int(args.count * args.max_repo_fraction))
    over = {k: v for k, v in by_repo.items() if v > ceiling}
    for repo, n in sorted(over.items()):
        # Not a bug and not silent: with 36 usable instances a draw of 30 takes
        # most of the pool, so a repo that dominates the pool must dominate the
        # draw. Saying so beats a cap that quietly did nothing.
        in_pool = sum(1 for r in pool if repo_of(r["instance_id"]) == repo)
        print(f"  note  : {repo} is {n}/{len(chosen)} of the draw, over the "
              f"{ceiling} ceiling -- it is {in_pool}/{len(pool)} of the pool "
              f"and the draw is too large a fraction of it to avoid")
    if dropped:
        print("  excluded:")
        for row, why in dropped:
            print(f"    {row['instance_id']:34} {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
