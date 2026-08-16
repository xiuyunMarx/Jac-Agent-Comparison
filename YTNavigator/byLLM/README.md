# byLLM YT-Navigator

The Jac/byLLM counterpart of the LangGraph chat agent in `../YT-Navigator`.
Same models, same routing decision, same tools, same database, same structured
answer shape — implemented with Jac's object-spatial graph and `by llm`
functions instead of LangGraph's StateGraph.

## Mapping to the original

| Original (LangGraph) | Counterpart (byLLM) |
|---|---|
| `StateGraph` router → 3 conditional edges (`main_graph.py`) | `root -> Router -> {StaticReply, DirectReply, ToolReply}` OSP graph; `ChatAgent` walker traversal (`nodes.jac`) |
| `ROUTE_QUERY_SYSTEM_PROMPT` + `PydanticOutputParser(AgentRouterOutput)` | `def route_query(...) -> Route by llm` — enum return, prompt carried by `sem` |
| ReAct subgraph on `qwen-qwq-32b` (`react_graph.py`) | `def tool_reply(...) by llm(tools=[...])` — byLLM's built-in ReAct loop |
| `AgentOutput` / `AgentOutputVideos` pydantic schemas | `AgentAnswer` / `AnswerVideo` objs — structured output native to `by llm` |
| `similarity_videos_search` + `execute_query` StructuredTools | Same two tools in `tools.jac`, plumbing in `retrieval.py` |
| Router parse fallback / error fallback (silent) | Same fallbacks, recorded as `fallback_events` |
| LangSmith/Postgres checkpointing | Not needed: conversation handled per-run by the benchmark runner |

Models default to the same pair as the original app (`gpt-4o-mini` for
routing and direct replies, `gpt-4o` for the tool agent), overridable with
the same `INSTANT_LLM` / `POWERFUL_LLM` env vars — langchain-style
`provider:model` values are normalized to litellm's `provider/model`, so one
env setting configures both implementations. Temperature 0 everywhere.

**Documented divergences** (kept intentionally small):
- Keyword search is a dependency-free BM25 (Okapi) reimplementation instead of
  `langchain BM25Retriever` (same algorithm family, same top-k=4).
- The cross-encoder reranking step of the original search tool is not
  reproduced; hits are ordered by cosine similarity and video citation count.
- Conversation trimming is "last 3 exchanges" instead of a 1000-token budget.

## Files

- `nodes.jac` — the agent: models, `Route` enum, answer schema, the three
  `by llm` functions, graph nodes, `ChatAgent` walker.
- `tools.jac` — LLM-facing tools with call logging + the litellm token/latency
  tracker (per-call records, shared by all `by llm` calls).
- `retrieval.py` — framework-neutral DB plumbing (pgvector semantic search,
  BM25, SQL tool, channel info) against the exact tables the original app
  populates.
- `main.jac` — headless benchmark runner: reads the shared questions JSONL,
  writes shared-schema results JSONL (one record per question: route, answer,
  cited videos, tool calls, per-LLM-call tokens/latency, fallback events).

## Setup

```bash
jac install       # from this directory - deps are declared in jac.toml
```

The `jac` runtime has its own Python environment: packages pip-installed into
your shell's env are NOT visible to it, so always use `jac install`.

Environment (same names as YT-Navigator's `.env`): `POSTGRES_*`,
`OPENAI_API_KEY`, optional `INSTANT_LLM` / `POWERFUL_LLM`, plus the runner's
`YTNAV_QUESTIONS` / `YTNAV_OUTPUT` / `YTNAV_CHANNEL`. Set
`YTNAV_FAKE_EMBEDDINGS=1` only for smoke runs against a dataset built with
`--fake-embeddings`.

The database must already contain a channel — build the synthetic benchmark
dataset (`python ../datasets/build.py`) or load a snapshot from the original
app (`python manage.py benchmark_snapshot load <snapshot> --replace`).

## Run

```bash
jac run main.jac                      # standalone (env vars set manually)
python ../eval/run.py --impl byllm    # via the comparison harness
```
