# Case studies

Turns graded runs from [`../swebench_bridge`](../swebench_bridge) into a study:
which implementations resolved what, where they disagreed, and what each one's
patch actually did to the tests.

```
build_study.py         graded run dirs -> verdicts.csv, divergence.csv, divergence.json, README.md
select_easy.py         the public leaderboard -> instances.txt, the set in use
select_instances.py    a divergence.csv -> instances-hard.txt, kept but unused
lite-01/               the byLLM vs LangGraph study, generated
instances.txt          the 8 instances the four-way run is pinned to
leaderboard_lite.json  the snapshot select_easy.py draws from
```

`instances.txt` holds **8 instances the public leaderboard says are easy**. The
36-instance hard set that used to live here — every instance one implementation
had already failed — was deleted: with four implementations to bring up, on a
local model, against a bridge and a container runtime that had never run
end to end, a set where everything fails cannot tell a broken container from a
weak agent. `select_instances.py` still generates that kind of set, now under
`instances-hard.txt`, for when there is something worth telling apart.

## Everything here is generated

The previous version of this directory was not. Its CSV, its JSON, its per-case
metadata and its whole status vocabulary existed only as checked-in files that no
code produced — which meant the numbers could not be re-derived, could not be
extended to the other 32 diverging instances, and could not be checked against
the runs they described. Nothing was wrong with them; there was just no way to
know that.

So:

```bash
python3 build_study.py --out lite-01 \
    ../swebench_bridge/results/lite-01-byllm \
    ../swebench_bridge/results/lite-01-langgraph
```

takes about three seconds, starts no container, calls no model, and rewrites
`lite-01/` from the run directories. Add a third run directory and it is a
three-way study; the generator does not count sides.

### Per-test status comes from the captured logs

`eval_logs/<id>/test_output.txt` plus the instance's own log parser recovers
FAIL_TO_PASS and PASS_TO_PASS exactly as the harness saw them. That is why a
study can be rebuilt from any graded run without re-grading it, and why
`divergence.json` can say *how* a patch failed rather than just that it did.

## The status vocabulary

One string per (instance, implementation). The two distinctions worth the extra
names:

| status | meaning |
|---|---|
| `RESOLVED` | FAIL_TO_PASS all pass, PASS_TO_PASS intact |
| `REGRESSION(P2P)` | every FAIL_TO_PASS passed **and then** a PASS_TO_PASS broke — the diagnosis was right and something came with it |
| `SUITE_ERROR` | no PASS_TO_PASS test passed at all: the run collapsed before it measured anything, usually an import error from unparseable source |
| `TESTS_FAIL` | the patch applied and did not fix it |
| `APPLY_FAIL` | the patch would not apply |
| `EMPTY_PATCH` | the run produced no patch |
| `HARNESS_ERROR` | grading itself failed; no verdict |

`REGRESSION(P2P)` is deliberately narrow. A patch that failed its own
FAIL_TO_PASS tests *and* broke others is just a wrong patch, and calling that a
regression would flatter it. `SUITE_ERROR` is separate because 0 of 862
matplotlib tests passing is not 862 regressions — nothing regressed, the suite
never ran.

## Choosing instances for the next run

```bash
python3 select_instances.py --count 30   # -> instances-hard.txt
```

Draws from `lite-01/divergence.csv`, because an instance every implementation
resolved — or none did — says nothing about the difference between them.

**This makes the set deliberately hard and non-representative.** Every instance
in it is one that at least one implementation already failed, so a resolve rate
measured on it is not comparable to a rate over the full 300. Any study built
from it has to say so, and the generated README does.

Two categories are refused as not-evidence, by rule rather than by hand:

- **split verdicts on byte-identical patches** — the same diff graded both ways
  is a flake in the harness, not a difference between the agents;
- **instances a side lost to infrastructure** — a container that never came up,
  or a provider 400. A *timeout* is not in this category and is kept: the agent
  spent its own budget and came back with nothing, which is a real result.

On `lite-01` those rules exclude exactly six of the 42 diverging instances, and
the draw is then balanced across the winning sides and capped per repo
(`--max-repo-fraction`, default 0.5 — django is 38% of Lite and over half the
divergence, and left alone would decide the comparison).

## The current study: `lite-01/`

byLLM vs LangGraph, SWE-bench Lite, all 300, gpt-4o. 63/300 and 65/300, 43
resolved by both, **42 diverging** — which is the interesting number, because it
means the two implementations agreed on only about two thirds of what they
individually got right.

See [`lite-01/README.md`](lite-01/README.md) for the tables, and
`lite-01/divergence.json` for the patches and per-test breakdowns.

## A set that is easy on purpose

A first end-to-end run should not be judged on instances chosen because someone
already failed them: if everything fails, nothing separates a broken container
from a weak agent. `select_easy.py` draws the opposite kind of set, from a prior
nobody here had to generate — **every SWE-bench Lite submission on the official
leaderboard**, at `SWE-bench/experiments/evaluation/lite/*/results/results.json`,
each listing exactly which of the 300 instances that system resolved.

```bash
python3 select_easy.py --count 8          # --refresh to re-fetch the snapshot
```

84 submissions have landed, and the spread is real: the top instances are
resolved by ~94% of them, and 35 of the 300 by none. The draw ranks on the
**recent window** (`--since`, default 2025 — systems built on models of roughly
the class we run) with the all-time rate as tie-break, applies a rate floor, and
caps any one repo at `--max-per-repo` (default 2) so django cannot supply the
whole set.

| instance | resolved by, 2025+ | all time |
|---|---|---|
| `django__django-11099` | 100% | 94% |
| `django__django-16255` | 100% | 94% |
| `sympy__sympy-13480` | 100% | 90% |
| `mwaskom__seaborn-3010` | 96% | 89% |
| `astropy__astropy-14995` | 96% | 82% |
| `scikit-learn__scikit-learn-13439` | 96% | 81% |
| `sympy__sympy-13471` | 96% | 63% |
| `pytest-dev__pytest-5227` | 93% | 88% |

All eight are one-file patches of 12–21 lines with 1–3 FAIL_TO_PASS tests, over
six repos. As with the hard set, **this is not a sample of the benchmark** — it
is the top of a difficulty ranking, so a rate measured on it is an upper bound
and is not comparable to a rate over the full 300. `leaderboard_lite.json` is the
snapshot it was drawn from, one line per submission, so the draw is reproducible
offline and `--check` fails if `instances.txt` drifts from the rules.

Five of the eight carry SWE-bench's own `difficulty` annotation, and all five
say `<15 min fix`; the other three are simply unannotated. That is an
independent check on the draw — the leaderboard prior and the human annotation
were produced by different people for different purposes and agree here.

## The run: four ways, one local model

`instances.txt`'s 8 instances, all four implementations, one model —
**muse-glimmer** (Meta's 30B agentic model, Apache 2.0) served locally by ollama
rather than a provider key, so the comparison costs GPU time instead of money.

```bash
cd ../swebench_bridge
export CODEAGENT_MODEL="openai/muse-glimmer"    # /v1, never ollama_chat/ -- see ../swebench_bridge/README.md
export OPENAI_BASE_URL="http://127.0.0.1:11435/v1" OPENAI_API_BASE="http://127.0.0.1:11435/v1"
export OPENAI_API_KEY="ollama" CODEAGENT_API_BASE="http://127.0.0.1:11435/v1"

python3 compare.py --run-id local-8 \
    --frameworks byllm langgraph openai nooa \
    --instances-file ../case_study/instances.txt \
    --runtime docker --model "$CODEAGENT_MODEL" \
    --python <an interpreter with langgraph, openai and nooa in it>

cd ../case_study
python3 build_study.py --out local-8 \
    ../swebench_bridge/results/local-8-{byllm,langgraph,openai,nooa}
```

`verdicts.csv` is then one row per instance with, per implementation, its status,
its token spend, its LLM calls and its wall clock — the per-case answer that a
resolve rate averages away.

## Tests

```bash
python3 -m pytest tests -q
```

Covers the status rules and the selection rules — the two places where this
directory turns run data into a claim.
