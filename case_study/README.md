# byLLM vs LangGraph — SWE-bench Lite case studies

Ten instances from run `lite-01` where exactly one of the two CodeAgent implementations produced a
patch that resolved the task. Each case is packaged as a standalone repair task: a repo checked out
at the buggy commit, the issue text, the reference fix, and a script that grades any candidate patch.

## Layout

```
<instance_id>/
  repo/                 the project at its buggy base commit (real git repo, branch `buggy`)
  problem_statement.md  the GitHub issue text — the only input an agent gets
  gold.patch            the reference fix
  test_patch.diff       the graded tests, applied on top of the base commit
  verify.sh             apply a patch, run the graded tests, print RESOLVED / NOT RESOLVED
  meta.json             repo, commit, image, test command, FAIL_TO_PASS / PASS_TO_PASS
```

## Verifying a fix

```bash
cd byllm_wins/django__django-12983
./verify.sh                  # gold.patch — should print RESOLVED
./verify.sh --none           # no fix — FAIL_TO_PASS must fail
./verify.sh /tmp/my_fix.diff # your agent's patch
```

By default `verify.sh` runs the official SWE-bench eval script inside the instance's
`swebench/sweb.eval.x86_64.*` image through `swebench_bridge/grade_local.py` and udocker; the image
is pulled on first use. `--local` runs the test command against `./repo` in the current shell, which
only works when the active environment already matches the instance (Python version included) —
`repo/` is there for the agent to read and edit, not to run everything natively.

`grade.py` (top level) parses a captured test log with swebench's own parser for that repo and
reports FAIL_TO_PASS / PASS_TO_PASS; `verify.sh --local` calls it for you.

## byLLM resolved, LangGraph did not

| case | bug | file to fix | F2P/P2P | why LangGraph's patch failed |
|---|---|---|---|---|
| [`astropy__astropy-12907`](byllm_wins/astropy__astropy-12907) | separability_matrix of nested CompoundModels | `astropy/modeling/separable.py` | 2/13 | LangGraph applied the fix to | composition too, 5 P2P regressions |
| [`django__django-12983`](byllm_wins/django__django-12983) | slugify strips dashes in the wrong order | `django/utils/text.py` | 1/15 | LangGraph stripped before the dash collapse instead of after |
| [`pytest-dev__pytest-7373`](byllm_wins/pytest-dev__pytest-7373) | remove the skipif/xfail evaluation cache | `src/_pytest/mark/evaluate.py` | 1/81 | LangGraph emitted unparseable Python (IndentationError) |
| [`scikit-learn__scikit-learn-13241`](byllm_wins/scikit-learn__scikit-learn-13241) | KernelPCA sign indeterminacy | `sklearn/decomposition/kernel_pca.py` | 1/54 | LangGraph flipped signs in transform() instead of at fit time |
| [`sympy__sympy-15609`](byllm_wins/sympy__sympy-15609) | LaTeX printing of MatrixElement indices | `sympy/printing/latex.py` | 1/121 | LangGraph patched _print_Indexed, not _print_MatrixElement, and deleted 1.9k test lines |

## LangGraph resolved, byLLM did not

| case | bug | file to fix | F2P/P2P | why byLLM's patch failed |
|---|---|---|---|---|
| [`django__django-11999`](langgraph_wins/django__django-11999) | cannot override get_FOO_display() | `django/db/models/fields/__init__.py` | 1/30 | byLLM had the right fix plus an extra edit causing infinite recursion |
| [`django__django-12700`](langgraph_wins/django__django-12700) | cleanse settings recursively in lists/tuples | `django/views/debug.py` | 1/77 | byLLM referenced an unbound name k -> NameError, 59 P2P errors |
| [`matplotlib__matplotlib-25311`](langgraph_wins/matplotlib__matplotlib-25311) | pickling a figure with a draggable legend | `lib/matplotlib/offsetbox.py` | 1/181 | byLLM patched DraggableLegend, not DraggableOffsetBox |
| [`pytest-dev__pytest-5692`](langgraph_wins/pytest-dev__pytest-5692) | hostname/timestamp in the JUnit XML | `src/_pytest/junitxml.py` | 2/68 | byLLM put imports inside the function and dedented the assignment |
| [`sympy__sympy-20442`](langgraph_wins/sympy__sympy-20442) | convert_to returns nonsense for some units | `sympy/physics/units/util.py` | 1/24 | byLLM changed an unrelated early return |

## Run totals

| | byLLM | LangGraph |
|---|---|---|
| Resolved | 63 / 300 (21.0%) | 65 / 300 (21.7%) |
| Empty patch | 18 | 12 |
| Harness error | 4 | 2 |

43 instances were resolved by both; 42 diverged. Five of those 42 are not capability differences and
were left out of this folder: `psf__requests-2317`, `sympy__sympy-24152` and `sympy__sympy-24213`
(byte-identical patches on both sides, verdicts split by harness flakes), plus `django__django-12184`
and `scikit-learn__scikit-learn-13439` (the loser submitted nothing because udocker failed to create
the container). `matplotlib__matplotlib-23314` is real but framework-level: LangGraph's run died on an
OpenAI 400 for an unanswered `tool_call_id`.

The three failure families across the ten cases: **unparseable edits** (pytest-7373 vs pytest-5692,
one each), **right diagnosis at the wrong site** (sklearn-13241, mpl-25311, sympy-15609, sympy-20442),
and **scope creep** — a correct fix plus a second edit that regresses PASS_TO_PASS (astropy-12907,
django-11999).

## Other files

- `divergence_all.csv` — all 42 diverging instances: winner, both verdicts, patch sizes, exclusions.
- `divergence_all.json` — the same rows plus the full gold / byLLM / LangGraph patches and per-test
  status. The agents' patches live here only; the case directories carry `gold.patch` alone.
- Run logs stay in `swebench_bridge/results/lite-01-{byllm,langgraph}/{logs,eval_logs}/<instance_id>/`.
