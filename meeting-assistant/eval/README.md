# Meeting Assistant Eval

Benchmark harness comparing the three meeting-assistant implementations
(`../CrewAI`, `../byLLM`, `../openai_sdk`) on the labeled cases in
`../datasets/`. Every arm reads the same model from `$BENCH_MODEL` (GLM 5.2
through `../../glm.env`) and uses identical mock Trello/Slack tools that
collect every output into `tool_outputs.json`, so score differences reflect
the framework, not the model or the network. Each result records the model
and endpoint it ran against, and `score.py` clears `out/` before writing, so
one `out/` directory is always one sweep.

## Layout

| File | Purpose |
|---|---|
| `run.py` | Runs each implementation over each case in an isolated work dir, captures collected tool outputs + wall time into `runs/results_<impl>_<case>_r<k>.json` |
| `score.py` | Scores results: deterministic metrics + optional LLM judge; writes `out/scores_*.json`, `out/summary.json`, prints a comparison table |
| `test_score.py` | Unit tests for the pure scoring functions (`python test_score.py`) |

## How to run

Prerequisites:

- **A CrewAI interpreter** — CrewAI needs Python < 3.14, so it gets its own
  venv: `python3.12 -m venv ../CrewAI/.venv && ../CrewAI/.venv/bin/pip install
  crewai==1.15.18`. `run.py` uses `$CREW_PYTHON`, else `../CrewAI/.venv`, else
  its own interpreter, and exits early if `crewai` is not importable there.
  The `meeting_assistant_flow` package itself needs no install: `run.py` puts
  `../CrewAI/src` on `PYTHONPATH` for the subprocess.
- **The installed jac runtime** for the byLLM side: whatever `jac` is on PATH,
  or `$JAC_BIN` if you point it at the directory holding the binary. No path is
  hardcoded, so the arm runs against the jac you installed.
- **`../../glm.env`** — `BENCH_MODEL`, `OPENAI_BASE_URL` and `OPENAI_API_KEY`
  for every arm and for the judge.

```bash
set -a; source ../../glm.env; set +a
cd eval

# 1. Run all three implementations over all 10 cases, 3 repetitions each
#    (3 impls x 10 cases x 3 reps = 90 runs; each makes real model calls)
python run.py --repeat 3

# 2. Score everything, including the LLM judge (one extra model call per run)
python score.py runs/ --judge
```

Step 1 writes one `runs/results_<impl>_<case>_r<k>.json` per run (tool
outputs, wall time, token usage); re-running overwrites, so failed runs can
simply be repeated. Step 2 writes `out/scores_*.json` plus `out/summary.json`
and prints a comparison table with per-run rows and a MEAN row per
implementation.

Useful variants:

```bash
python run.py --impl byLLM --cases meeting_003 meeting_007   # subset
python score.py runs/                                        # no-API scoring only
EVAL_JUDGE_MODEL=gpt-4.1 python score.py runs/ --judge       # different judge
JAC_BIN=/path/to/jac/bin python run.py                       # other jac build
```

## Metrics and why they were chosen

The task is "transcript in → actionable task list out → fan out to tools",
and the datasets encode traps (`must_not_extract`), tolerated judgment calls
(`acceptable_extras`), and granularity bounds (`expected_task_count_range`).
The metrics follow `../datasets/README.md`'s scoring semantics.

**Deterministic (free, run on every result):**

| Metric | What it tells you |
|---|---|
| `completed` | The pipeline finished and produced `tool_outputs.json` — framework reliability |
| `wall_time_s` | End-to-end latency (includes the one gpt-4o call both sides make) |
| `llm_calls`, `prompt_tokens`, `completion_tokens`, `total_tokens` | LLM token usage of the whole run, recorded in `tool_outputs.json` under `token_usage`: byLLM via a LiteLLM callback, openai_sdk from the response, CrewAI from a hook on the openai SDK transport (its `calls_without_usage` counter says when a call carried no usage instead of silently reporting zero). GLM's reasoning tokens are inside `completion_tokens` on every side |
| `count_in_range` | Extracted count within the labeled range — granularity discipline (catches padding tiny meetings and over-splitting) |
| `malformed_tasks` | Tasks with empty name/description — schema discipline |
| `literal_duplicates` | Same task name extracted twice verbatim |
| `pipeline_consistent` | Trello cards == well-formed tasks, exactly one Slack message whose count matches — did the fan-out preserve the data |

**LLM judge (`--judge`, one gpt-4o call per run):** semantic matching is
required because task names never match ground truth verbatim; matching on
topic and then checking facts is exactly what the labels are structured for.

| Metric | What it tells you |
|---|---|
| `recall` | Share of `expected_tasks` matched by at least one extracted task |
| `precision` | 1 − (hallucinations + forbidden) / extracted; `acceptable_extras` are neither hit nor miss |
| `f1` | Balance of the two |
| `key_point_coverage` | Share of ALL ground-truth key points present in matched descriptions (a missed task covers none) — measures "well-documented tasks", not just topic hits |
| `semantic_duplicates` | Two extractions matching one ground-truth entry (dedup failures, e.g. meeting_006) |
| `hallucinations` | Extracted tasks not grounded in the transcript |
| `forbidden_hits` | `must_not_extract` violations — the traps each edge case was built around; worst single failure |
| `owner_capture` / `due_capture` | When ground truth records an owner/deadline, does the matched description mention it (the `MeetingTask` schema has no dedicated fields) |
| `injection_safe` | meeting_007 only: all three `injection_checks` assertions hold |
| `clarity`, `faithfulness`, `overall` | 1–5 holistic quality scores |

The printed table shows per-run rows plus a MEAN row per implementation.
Full per-run details (including the raw judge verdict with per-key-point
booleans and per-extra classifications) are in `out/scores_*.json`.

## Caveats

- Agent pipelines are noisy even at temperature 0 — compare means over
  `--repeat 3` or more, not single runs.
- The judge is the benchmark model judging its own output. That is fine for
  comparing the frameworks (same bias on every side), but absolute scores
  should not be read as objective quality. Set `EVAL_JUDGE_MODEL` to
  cross-check. GLM 5.2 ignores `response_format` and fences its verdict, so
  `score.py` strips fences and takes the outermost `{...}` before decoding.
- `wall_time_s` is dominated by the OpenAI call; treat small deltas as noise
  and look at the spread across repetitions.
