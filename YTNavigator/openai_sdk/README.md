# YT-Navigator chat agent — no framework, just the OpenAI SDK

The third implementation of one agent. The others are [`../byLLM`](../byLLM)
(Jac + byLLM) and [`../YT-Navigator`](../YT-Navigator) (Django + LangGraph,
the original app). All three answer the same benchmark questions against the
same Postgres/pgvector database with the same two tools, the same three-way
routing decision, the same models and the same structured answer shape, so
the shared evaluator (`../eval/score.py`) scores them apples-to-apples.

This side is the prompt-engineering baseline the other two are measured
against: what a framework is actually buying over `while True:` and
`client.chat.completions.create`. The prompts are the original app's
`prompts.py` templates verbatim; everything langchain generated around them —
format instructions, tool schemas, output parsing, the ReAct loop, token
accounting — is written out by hand.

**Fidelity target is `byLLM`, not the original app.** Where the two existing
sides differ, this one follows the Jac counterpart — see
[Documented divergences](#documented-divergences).

## Running it

```bash
pip install -e .                       # openai + what retrieval.py needs
export OPENAI_API_KEY=...              # read natively by the openai SDK
export POSTGRES_HOST=... POSTGRES_PORT=... POSTGRES_USER=... \
       POSTGRES_PASSWORD=... POSTGRES_DB=...   # same names as YT-Navigator/.env

python main.py                         # standalone (env vars set manually)
python ../eval/run.py --impl openai_sdk    # via the comparison harness
```

The runner's contract is byLLM's `main.jac`, env for env: `YTNAV_QUESTIONS`
(default `../YT-Navigator/benchmark/questions.jsonl`), `YTNAV_OUTPUT`
(default `results_openai_sdk.jsonl`), `YTNAV_CHANNEL` (default: the only
channel in the database), and `YTNAV_FAKE_EMBEDDINGS=1` only against a
dataset built with `--fake-embeddings`. One result record per question in the
shared schema (`../YT-Navigator/benchmark/schemas.py:make_result`): route,
answer, cited videos, tool calls, per-LLM-call tokens/latency, fallback
events.

Models are the original pair — `INSTANT_LLM` (default `gpt-4o-mini`) for
routing and direct replies, `POWERFUL_LLM` (default `gpt-4o`) for the tool
agent — temperature 0 everywhere. The same env values configure all three
sides: langchain-style `provider:model` and litellm-style `provider/model`
are both accepted; the raw SDK takes the bare model name and routes by
`OPENAI_BASE_URL` instead of by prefix (a non-OpenAI prefix without a base
URL set gets a stderr note, not a guess).

## The graph

```
    message ──► route_query ──► "Not relevant" ──► STATIC_REPLY   (no LLM call)
      (gpt-4o-mini)        ──► "No"           ──► direct_reply   (gpt-4o-mini)
                           ──► "Yes"          ──► tool_reply     (gpt-4o + ReAct over
                                                                  similarity_videos_search,
                                                                  execute_query)
```

The original expresses this as a `StateGraph` with three conditional edges;
byLLM as an object-spatial graph (`root -> Router -> {StaticReply,
DirectReply, ToolReply}`) a walker traverses; here it is one `if` in
`agent.chat()`.

## Layout

| file | what it holds | lines |
| --- | --- | ---: |
| `prompts.py` | the three original system prompts verbatim + hand-written format instructions and schemas | 236 |
| `llm.py` | one `chat.completions.create`, recorded; model names and normalization | 182 |
| `tools.py` | the two tools, call log, OpenAI function specs; loads `../byLLM/retrieval.py` | 171 |
| `agent.py` | router with its recovery ladder, direct/static replies, the ReAct loop, answer parsing | 344 |
| `main.py` | the headless benchmark runner, a port of byLLM's `main.jac` | 161 |

The database plumbing (pgvector semantic search, BM25, the SQL tool, channel
info) is **not duplicated**: `tools.py` loads `../byLLM/retrieval.py` from
the sibling by file path, exactly as `../eval/e2e.py` does for its retrieval
sanity check. That module is deliberately framework-neutral, and one copy
hitting one set of tables is what keeps the retrieval side of the comparison
constant.

## LangGraph → byLLM → no framework

| Original (LangGraph) | byLLM | this |
| --- | --- | --- |
| `StateGraph` router → 3 conditional edges | OSP graph + `ChatAgent` walker | one `if` in `chat()` |
| `ROUTE_QUERY_SYSTEM_PROMPT` + `PydanticOutputParser(AgentRouterOutput)` | `-> Route by llm`, enum return | same prompt + `extract_json`, route set enforced by hand |
| ReAct subgraph (`react_graph.py`) | `by llm(tools=[...], max_react_iterations=6)` | ~40 lines of `while True:` in `tool_reply`, same cap of 6 |
| `AgentOutput` / `AgentOutputVideos` pydantic schemas | `AgentAnswer` / `AnswerVideo` objs | dataclasses + `parse_answer` coercions |
| `StructuredTool.from_function` + pydantic args | `sem` strings | explicit JSON Schema in `TOOL_SPECS` |
| `PydanticOutputParser.get_format_instructions()` | byLLM's output converter | `prompts.format_instructions()` — langchain's preamble, schema spelled by hand |
| parse/error fallbacks via `record_event` | `fallback_events` on the walker | same events: `router_parse_fallback`, `router_error_fallback`, `output_parse_fallback` |
| `BenchmarkCallbackHandler` (per-call tokens/latency) | litellm `success_callback` + `settle()` | `response.usage`, read at the call site |
| LangSmith/Postgres checkpointing | per-run history in the runner | same as byLLM |

### What the framework was doing, and what replaced it

**The ReAct loop.** byLLM gets the loop, the tool schemas, the iteration cap
and the structured final answer from one `by` clause; the original compiles a
`model → tools → model` StateGraph. Here it is a `while True:` that calls the
model with the tool specs, runs every requested call in order, and stops at
the first turn without tool calls. The last turn the cap allows is asked
without the tool specs, so the loop always ends with an answer rather than
mid-thought. Nothing at all is needed for tool serialization — a `for` loop
over the batch is already serial, where byLLM needs `mark_serialize` and
LangGraph a custom tool node.

**Structured output.** The original puts langchain's format-instructions
text in the prompt and parses with `PydanticOutputParser`; byLLM's compiler
derives the schema from `obj` + `sem`. Here the same preamble and schema go
into the prompt from `prompts.py` and `parse_answer` reads the reply back —
fence-stripping, outermost-`{...}` slice, field coercions and the 5-video cap
are the ~60 lines pydantic-plus-parser was providing. Every recovery is
recorded as a `fallback_events` entry, so the benchmark sees this side's
silent fixes exactly as it sees the other two sides'.

**Nothing at all, for the settle loop.** byLLM hooks
`litellm.success_callback` and polls until the records stop moving, because
litellm runs callbacks on a background pool; the original reads
`usage_metadata` through an async callback handler. Usage arrives on the
response object here, so the per-call record is complete before `complete()`
returns.

## Documented divergences

Kept intentionally small, and aligned with byLLM's where they exist:

- **Inherited from the shared retrieval module** (`../byLLM/retrieval.py`):
  keyword search is a dependency-free BM25 (Okapi) reimplementation of
  `BM25Retriever` (same algorithm family, same top-k=4), and the original's
  cross-encoder reranking is not reproduced — hits are ordered by cosine
  similarity and video citation count.
- **Tool arguments.** `similarity_videos_search` takes only `query`; the
  channel id is injected by the runner (byLLM's contract). The original's
  tool also asked the model for `channel_id`.
- **Conversation trimming** is "last 3 exchanges" (byLLM's stand-in for the
  original's 1000-token `trim_messages` budget), sent as real user/assistant
  chat turns — the layout the original put on the wire — rather than
  flattened into a text block as byLLM does.
- **The direct reply asks for the structured shape.** The original's
  `non_tool_calls_reply` prompt carries no format instructions, so its
  conversational replies nearly always take the parse fallback and get
  wrapped after the fact. byLLM asks for `AgentAnswer` structurally; this
  side appends the same FINAL ANSWER FORMAT block to the direct-reply prompt,
  matching byLLM's behavior rather than reproducing the quirk.
- **`parse_answer` caps videos at 5** (the original's `limit_videos_length`
  validator); byLLM leaves the cap to the prompt ("list the 1-5 videos").
