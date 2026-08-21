# YT-Navigator Agent Benchmark

A headless, reproducible harness for benchmarking the YT-Navigator chat agent —
and for comparing alternative implementations of the same agent (e.g. the
LangGraph version in this repo vs. a Jac/Jaseci re-implementation) against
identical data, identical questions, and identical metrics.

## How it differs from the interactive app

| Interactive app | Benchmark mode |
|---|---|
| Reached through Django views + auth | `manage.py benchmark_run` invokes the graph directly |
| Chat history keyed by user id, persists across sessions | Fresh conversation thread per question (or per scenario), deleted after the run |
| Live YouTube scraping | Frozen dataset snapshot, loaded with `benchmark_snapshot` |
| Every interaction pushed to LangSmith in a background thread | LangSmith collection disabled (`BENCHMARK_MODE`) |
| Errors silently recovered (router parse fallback, rerank fallback, ...) | Same recovery, but every fallback is recorded as an event and reported as a metric |
| Router LLM at default temperature | Both LLMs at temperature 0 |

## Workflow

```bash
# 1. One-time: scan a channel through the web UI as usual, then freeze the data.
python manage.py benchmark_snapshot dump benchmark/data/snapshot.json.gz

# (on any other machine/database: restore it)
python manage.py migrate
python manage.py benchmark_snapshot load benchmark/data/snapshot.json.gz --replace

# 2. Author a question set (copy questions.example.jsonl and fill in real
#    video ids / reference answers for the channel you snapshotted).

# 3. Run the agent headlessly.
python manage.py benchmark_run benchmark/questions.jsonl \
    --channel <CHANNEL_ID> -o benchmark/results/langgraph.jsonl

# 4. Score one run, or compare several implementations side by side.
python benchmark/evaluate.py benchmark/results/langgraph.jsonl \
    benchmark/results/jac.jsonl \
    --questions benchmark/questions.jsonl --judge --report benchmark/results/report.json
```

`benchmark_run` requires the usual app environment (Postgres reachable,
`OPENAI_API_KEY` set — or the key for whatever provider `INSTANT_LLM` /
`POWERFUL_LLM` point at). It does **not** require `LANGSMITH_API_KEY`.

## Question format (`questions.jsonl`)

One JSON object per line:

```json
{"id": "q2",
 "question": "Which video explains transformers?",
 "expected_route": "Yes",
 "expected_video_ids": ["dQw4w9WgXcQ"],
 "reference_answer": "The video 'Attention is all you need, explained' covers ...",
 "scenario_id": null}
```

- `expected_route` — the correct router decision: `"Yes"` (needs tools),
  `"No"` (direct reply), `"Not relevant"` (refusal). Used for routing accuracy.
- `expected_video_ids` — videos a correct answer should cite. Used for
  retrieval hit rate / recall.
- `reference_answer` — gold answer, used only by the optional LLM judge.
- `scenario_id` — questions sharing a `scenario_id` run in one conversation
  thread in file order (multi-turn tests). All other questions are isolated.

## Result format (`results.jsonl`)

One record per question, defined canonically in `schemas.py:make_result`. Key
fields: `route`, `answer_text`, `cited_video_ids`, `tool_calls`, `llm_calls`
(per-call model/latency/tokens), `fallback_events`, `latency_s`,
`prompt_tokens` / `completion_tokens`, `status` / `error`.

**Contract for other implementations (e.g. Jac):** run the same questions file
against the same snapshot and write records with these same fields
(`framework` set to your label). `evaluate.py` will then score both files in
one table. The models (`INSTANT_LLM`, `POWERFUL_LLM`), prompts, temperature
(0), and dataset must match the reference implementation for the comparison to
be meaningful.

## Metrics reported by `evaluate.py`

- **Completed / error rate** — invocations that returned vs. raised.
- **Fallback rate** — runs where the agent silently recovered from an internal
  failure (router output unparsable, reranker failure, ...). In the app these
  look like normal answers; here they count against the implementation.
- **Routing accuracy** — router decision vs. `expected_route`.
- **Retrieval hit rate / recall** — cited vs. `expected_video_ids`.
- **Answer parse rate** — final answers conforming to the structured
  `AgentOutput` schema.
- **Judge score** (`--judge`) — 1–5 factual agreement with `reference_answer`,
  graded via litellm (default `gpt-4o-mini`; keep the judge model different
  from — ideally stronger than — the models under test).
- **Latency mean / p50 / p95, tokens per question, LLM/tool calls per question.**

## Reproducibility checklist

- [ ] Same snapshot loaded (`benchmark_snapshot load ... --replace`)
- [ ] Same question file
- [ ] Same `INSTANT_LLM` / `POWERFUL_LLM` env values on every run
- [ ] `OPENAI_API_KEY` set; `LANGSMITH_API_KEY` not needed
- [ ] Don't run benchmark and interactive chat against the same database
      concurrently

Note: temperature 0 makes runs *more* stable, not bit-identical — hosted LLM
inference is not perfectly deterministic. For publishable numbers, run each
question set several times and report the spread.
