# Case studies

Turns graded runs from [`../swebench_bridge`](../swebench_bridge) into a study:
which implementations resolved what, where they disagreed, and what each one's
patch actually did to the tests.

```
build_study.py       graded run dirs -> verdicts.csv, divergence.csv, divergence.json, README.md
select_instances.py  a divergence.csv -> instances.txt, for the next comparison run
lite-01/             the byLLM vs LangGraph study, generated
instances.txt        the 30 instances pinned for the three-way run
```

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
python3 select_instances.py --count 30
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

## The next study: three ways

`instances.txt` holds the 30 instances selected above. Nothing has been run
against them yet; `openai_sdk` has never been run against SWE-bench at all.

```bash
cd ../swebench_bridge
python compare.py --run-id three-way --frameworks byllm langgraph openai \
    --instances-file ../case_study/instances.txt --model gpt-5

cd ../case_study
python3 build_study.py --out three-way \
    ../swebench_bridge/results/three-way-byllm \
    ../swebench_bridge/results/three-way-langgraph \
    ../swebench_bridge/results/three-way-openai
```

That run bills a provider key: on the gpt-5 evidence available (~865k tokens and
~800s per instance) 30 instances × 3 implementations is on the order of 78M
tokens and several hours. Smoke-test one instance first — `openai_sdk` has no
test suite of its own and has never completed an LLM round trip under the
harness:

```bash
python run_agent.py --framework openai --run-id smoke \
    --instance-ids astropy__astropy-12907 --model gpt-5 --workers 1
```

## Tests

```bash
python3 -m pytest tests -q
```

Covers the status rules and the selection rules — the two places where this
directory turns run data into a claim.
