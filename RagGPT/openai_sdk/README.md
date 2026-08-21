# Jac-GPT — no framework, just the OpenAI SDK

The fourth implementation of one RAG chatbot. The others are
[`../Jac-Rag-GPT`](../Jac-Rag-GPT) (Jac + byLLM, graph router),
[`../Jac-Rag-GPT-ByllmRouter`](../Jac-Rag-GPT-ByllmRouter) (Jac + byLLM,
classification router), and [`../langgraph`](../langgraph) (Python +
LangGraph). All four are scored by the shared eval at [`../eval`](../eval), so
they must present the same shape: the same five agents with **verbatim
identical prompts**, the same single `search_docs` tool into the same FAISS
corpus, the same session/turn contract, the same `gpt-4.1-mini`.

This side is the prompt-engineering baseline the others are measured against:
what a framework is actually buying over `client.chat.completions.create`,
`json.loads`, and a `for` loop.

**Fidelity target is `langgraph`, not the Jac sides.** The eval's known
asymmetries (router temperature, router history, fallback behaviour) already
distinguish the three existing systems; this side takes the LangGraph position
at each of those points — its prompts.py is where the shared strings were
transcribed, its process/driver contract is what this side plugs into, and its
`faiss_index/` is byte-copied here.

## Running it

```bash
pip install -e .
export OPENAI_API_KEY=...        # read natively by the openai SDK; ../.env also works
export MODEL=gpt-4.1-mini        # optional; same default and override hook as the other sides

python jac-gpt.py                # terminal chat, same REPL as ../langgraph/jac-gpt.py
```

Under the eval, `OPENAI_BASE_URL` points every LLM call — router, ReAct
iterations, retries — at the token-counting proxy. The raw `openai` client
honors that variable natively, so this side needs no code for it.

## Layout

| file | what it holds |
| --- | --- |
| `jac-gpt.py` | router, ReAct loop, sessions, `JacGPTFactory`, CLI |
| `prompts.py` | the five agent prompts + router intent (verbatim), tool schema, router schema |
| `rag_engine.py` | config parse, index load (restricted unpickler), search, rerank, summary |
| `config/faiss_reranking.json` | copy of the shared config (same values on every side) |
| `faiss_index/` | byte-copy of `../langgraph/faiss_index` — identical vectors and chunks |
| `docs` | symlink to `../langgraph/docs`; runtime never reads it (see divergences) |

## byLLM → LangGraph → no framework

| byLLM (Jac) | LangGraph | this |
| --- | --- | --- |
| `node RagChat` … + `sem` strings | `AgentSpec` in prompts.py | same dataclass, same strings, minus pydantic |
| `visit [-->] by llm(select=1, intent=…)` | `with_structured_output(RouteDecision)` | one completion with a `json_schema` response format over the five names |
| `def respond(...) by llm(tools=[search_docs])` | `create_react_agent(...)` | `JacGPT.react`, a `for` loop |
| `max_react_iterations=N` | `recursion_limit = 2N + 1` | `range(N + 1)`, break if the last call still wants tools |
| *(byllm runtime message)* | `"Stopped after N tool iterations…"` on `GraphRecursionError` | the same string, returned by the loop itself |
| `sem search_docs` | `@tool(description=…)` + pydantic args | `SEARCH_DOCS_TOOL`, the wire-format dict written out |
| `node Session` on the graph | `SessionInfo` + factory dict | same `SessionInfo` + factory dict |
| `rag_engine.jac` over langchain FAISS | `rag_engine.py` over langchain FAISS | `rag_engine.py` over `faiss` + `sentence-transformers` directly |

### What the framework was doing, and what replaced it

**The ReAct loop.** byLLM gets it from one `by llm(tools=…)` clause; LangGraph
from `create_react_agent`. Here it is ~30 lines in `JacGPT.react`: call the
model with the one tool schema, execute and append every requested call,
repeat. The two behaviours worth care are the iteration budget (mapped from
LangGraph's `recursion_limit = 2N+1`: N tool batches plus one final model
call) and what happens when it runs out — the same canned "Stopped after N
tool iterations" answer LangGraph's `GraphRecursionError` handler produced,
because the eval judges that string, not an exception.

**The router.** `with_structured_output(RouteDecision)` becomes one
`chat.completions.create` with a `response_format` json_schema whose only
property is an enum of the five agent names — same intent text, temperature 0,
no chat history, and the same fall-back-to-RagChat on any failure.

**The retrieval stack.** LangChain's FAISS wrapper was fronting four lines of
query-time work: encode the query with the local `BAAI/bge-small-en-v1.5`
model (no API involved), `index.search` for the top 30 by L2 distance, map row
ids to docstore ids, look up the documents. `rag_engine.py` does exactly that
against a byte-copy of the LangGraph side's index, then reranks with the same
CrossEncoder call (same model, batch size, top-7 cut) and renders the same
`{source} (Relevance Score: {score:.4f}) : {original_content}` summary lines.
Loading the index means reading langchain's `index.pkl`, which pickles two
langchain classes; a restricted unpickler maps exactly those two onto local
stand-ins and refuses every other class.

## The eval harness

Registered in `../eval/common.py` as `openai-sdk`, kind `python` — one added
line. No driver changes: the harness's in-process Python driver execs
`<dir>/jac-gpt.py` and calls `JacGPTFactory().interact(message=, session_id=)`,
and this side exports exactly that file and contract. Because that driver may
load both Python systems into one process, `jac-gpt.py` imports its own
`prompts.py`/`rag_engine.py` by absolute path under `openai_sdk_`-prefixed
module names rather than bare `import prompts`, which would hit whichever
side's module was cached first.

```bash
/home/xiaoyu/miniconda3/envs/jaseci/bin/python ../eval/harness/run_eval.py \
    --systems openai-sdk --limit 5 --repeats 1     # smoke; drop the flags for the full run
```

## Divergences from the siblings

1. **No index builder.** The Jac and LangGraph sides rebuild `faiss_index/`
   from `docs/` when it is missing, via langchain's loaders and
   `RecursiveCharacterTextSplitter`. Reimplementing that chunker invites
   silent corpus drift, so this side only loads; a missing index raises with a
   pointer at the langgraph builder. The shipped index is a byte-copy of
   `../langgraph/faiss_index`, which is the strongest available parity: both
   sides search the same vectors over the same chunks. (Chunk metadata inside
   it records the pre-rename `…/Coder/langgraph/docs/…` source paths; the
   LangGraph side serves those same strings, and the RagChat prompt tells the
   model to cite section names, not paths.)
2. **`docs` is a symlink** to `../langgraph/docs` rather than a fourth 2.7 MB
   copy, since without a builder the runtime never reads it. The other three
   systems each carry a full copy.
3. **Config is parsed properly.** The nested `config/faiss_reranking.json` is
   read the way the Jac sides read it. The LangGraph side feeds the nested
   dict to a flat dataclass, always fails, and lands on defaults that happen
   to equal the file (a known asymmetry noted in the eval README) — same
   effective values on every side either way.
4. **Router wire format.** byLLM routes by `select=1` over graph edges (and
   the ByllmRouter variant by enum classification); LangGraph routes by
   pydantic structured output; this side by a `json_schema` response format.
   Same prompt text, same temperature 0, same five-way choice — the mechanism
   is the measured variable.
5. **One OpenAI client per factory**, shared across sessions, instead of
   LangGraph's six `ChatOpenAI` instances per session. The eval creates
   hundreds of sessions per run; per-session connection pools are pure
   overhead with no behavioural surface.
