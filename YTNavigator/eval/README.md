# YT-Navigator: LangGraph vs byLLM evaluation

Runs both implementations of the YT-Navigator chat agent over the same
questions and the same database, and scores them side by side. Follows the
layout of the sibling comparisons (`Email-Auto-response/eval`,
`meeting-assistant/eval`).

- `../YT-Navigator` — original Django + LangGraph implementation
  (headless via `manage.py benchmark_run`)
- `../byLLM` — Jac/byLLM counterpart (headless via `jac run main.jac`)
- `../datasets` — synthetic benchmark channel + 23 ground-truth questions
- shared contract & scorer — `../YT-Navigator/benchmark/` (`schemas.py`, `evaluate.py`)

## One command

```bash
export OPENAI_API_KEY=...      # or put it in ../YT-Navigator/.env
python e2e.py                  # DB -> dataset -> both agents -> score table
```

`e2e.py` stages: prereq checks → database (uses `POSTGRES_*` if reachable,
else starts a dockerized `pgvector/pgvector:pg16` on port 5544) → build the
synthetic dataset → retrieval sanity check → run byLLM → run LangGraph (auto-
skipped with a note if its Python env isn't available) → score.

Useful variants:

```bash
python e2e.py --smoke --fake-embeddings   # validate everything except LLM calls; no key, no torch
python e2e.py --impl byllm                # one side only
python e2e.py --judge                     # add 1-5 LLM-judge scoring vs reference answers
python e2e.py --langgraph-python ~/venvs/ytnav/bin/python
python e2e.py --replace-data              # rebuild the dataset first
```

## Per-implementation environments

- **byLLM**: dependencies are declared in `../byLLM/jac.toml` — run
  `cd ../byLLM && jac install` once. Note: the `jac` runtime has its own
  Python; packages pip-installed into your shell's env are NOT visible to it.
- **LangGraph**: needs a Python env with the app installed
  (`pip install -e ../YT-Navigator`) plus `sentence-transformers`; pass its
  interpreter via `--langgraph-python`. On a fresh database, e2e runs
  `makemigrations` + `migrate --fake-initial` for it automatically.
- **The driver env** (whatever runs `e2e.py`): `psycopg2-binary`, and
  `sentence-transformers` for building real embeddings.

## No docker? Automatic conda-Postgres fallback

When docker is unusable, `e2e.py` looks for Postgres binaries that ship the
pgvector extension (on PATH or in any conda env) and manages a local instance
itself: data in `eval/pgdata`, port 5544, kept running between runs (stop with
`pg_ctl -D eval/pgdata stop` from that env's bin). If none exist, one command
provides them — no root needed:

```bash
conda create -y -n ytnav-pg -c conda-forge postgresql pgvector   # then rerun e2e.py
```

## Outputs

`out/results_<impl>.jsonl` (shared schema, one record per question) and
`out/report.json`. Re-score without re-running:

```bash
python score.py out/results_byllm.jsonl out/results_langgraph.jsonl \
    --questions ../datasets/questions.jsonl
```

## Metrics

Routing accuracy, retrieval hit rate/recall (cited vs expected videos),
structured-output parse rate, silent-fallback rate, error rate, latency
mean/p50/p95, tokens per question, LLM/tool calls per question, optional
judge score. Definitions in `../YT-Navigator/benchmark/README.md`.

Agent pipelines are noisy even at temperature 0 — run each side several times
and compare means before drawing conclusions.
