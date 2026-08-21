# Jac-GPT three-system eval: task quality + token usage

Compares `langgraph/`, `Jac-Rag-GPT/`, and `Jac-Rag-GPT-ByllmRouter/` on a
docs-grounded synthesized dataset. The five agent prompts are verbatim
identical across the systems, so differences trace to the router mechanism
and framework overhead.

## Prerequisites

1. `OPENAI_API_KEY` in `Coder/.env` (one line: `OPENAI_API_KEY=sk-...`).
   Needed by the systems (gpt-4.1-mini), synthesis and judging (gpt-4.1).
2. **The installed jac runtime.** Scripts use whatever `jac` is on PATH, or
   `$JAC_BIN` if you set it to the directory holding the binary; no path is
   hardcoded. Run `jac install` once in each Jac project (`Jac-Rag-GPT`,
   `Jac-Rag-GPT-ByllmRouter`) under that same runtime -- packages pip-installed
   into your shell's environment are not visible to it.
3. Python: the environment built from the repo root's `requirements.txt`.

## Pipeline

```bash
PY=/home/xiaoyu/miniconda3/envs/jaseci/bin/python

# 1. Synthesize the dataset (~220 items; resumable; --smoke for 3/category)
$PY dataset/synthesize.py

# 2. Run the eval (starts the token proxy itself; resumable)
$PY harness/run_eval.py --repeats 3            # full run
$PY harness/run_eval.py --limit 5 --repeats 1  # smoke test first

# 3. Score (routing + jac check + gpt-4.1 judge; judge calls cached)
$PY harness/score.py

# 4. Aggregate
$PY harness/report.py                          # -> results/report.md + CSVs
```

## How token counting works

`harness/proxy.py` is a logging reverse proxy on `127.0.0.1:8899` in front of
`api.openai.com`. Every system process runs with
`OPENAI_BASE_URL=http://127.0.0.1:8899/v1`, which litellm (byllm) and
langchain-openai both honor, so all LLM calls — router, ReAct iterations,
retries — are logged with token usage under a per-turn marker the runner sets
via `POST /__mark`. This is neutral and identical instrumentation for all
three systems; byllm's own non-streaming path discards usage internally, so a
proxy is the only non-invasive way to count.

Judge/synthesis traffic goes direct to `api.openai.com` and never pollutes
the proxy log.

## Dataset (dataset/dataset.jsonl)

| category | n | gold agent | gold artifact | scoring |
|---|---|---|---|---|
| rag_qa | 80 | RagChat | verified gold answer + doc section | judge 1-5 vs gold |
| coding | 40 | CodingChat | reference that passes `jac check` | compile rate + judge |
| debugging | 40 | DebuggerChat | mutant fails / fix passes `jac check` | compile rate + judge |
| small_talk | 20 | QAChat | — | binary appropriateness |
| off_topic | 20 | OffTopicChat | — | binary appropriateness (redirects?) |
| multi_turn | 20 | per turn | turn 2 unroutable without history | turn-2 routing accuracy |

Sources: the ~26 substantive doc files under `Jac-Rag-GPT/docs/` (release
notes, fullstack/production tutorials, the syntax cheatsheet, and
breaking-changes are excluded).

## Files

- `common.py` — paths, env setup, `jac_check()`, jsonl helpers
- `dataset/synthesize.py` — dataset builder (gpt-4.1, resumable)
- `harness/proxy.py` — token-counting proxy
- `harness/drivers.py` — `jac start` REST driver + in-process LangGraph driver
- `harness/run_eval.py` — sequential runner, resume unit = (system, item, repeat)
- `harness/score.py` — routing/compile/judge scoring -> `results/judged.jsonl`
- `harness/report.py` — tables, bootstrap CIs, Wilcoxon -> `results/report.md`
- `harness/jac_check_env/` — scratch Jac project for compile checks

## Known asymmetries (by design — the eval measures them, report notes them)

- `Jac-Rag-GPT` routes at byllm's default temperature 0.7; the other two at 0.
- The LangGraph router never sees chat history; ByllmRouter feeds the last 10
  messages; Jac-Rag-GPT passes implicit walker context only.
- Router fallback: LangGraph -> RagChat on exception; ByllmRouter -> OffTopicChat
  structurally; Jac-Rag-GPT has none.
- LangGraph's config parsing silently falls back to dataclass defaults, which
  happen to equal `config/faiss_reranking.json`.
