# Jac-Rag-GPT-ByllmRouter (Jac-GPT core, byllm-router variant)

Same engine as [Jac-Rag-GPT](../Jac-Rag-GPT/) — a RAG-grounded multi-agent
chatbot over the bundled Jac docs — but with a different routing mechanism:
instead of delegating node selection to the graph traversal itself
(`visit [-->] by llm(select=1, intent=...)`), a `by llm` **classification
function** returns an `AgentType` enum and the walker visits the matching
agent node explicitly.

## Architecture

```
interact walker (message, session_id)
  └─ Root: find/create Session, record user turn
      └─ Router: choice = select_agent(message, history)   ← byllm enum classifier
                 if/elif on choice → visit [-->][?:<Agent>]
          ├─ RagChat       docs Q&A       (ReAct + search_docs tool)
          ├─ CodingChat    write Jac code (ReAct + search_docs tool)
          ├─ DebuggerChat  fix Jac code   (ReAct + search_docs tool)
          ├─ QAChat        small talk
          └─ OffTopicChat  polite redirect
  └─ Root exit: save assistant turn, report {session_id, agent, response}
```

The routing prompt lives in `main.impl.jac` as semstrings: `sem select_agent`
carries the routing instructions and `sem AgentType.<MEMBER>` describes each
agent — byllm feeds these to the LLM as the output schema, replacing the
`intent=ROUTER_INTENT` string of the original.

- `main.jac` — agent nodes, `AgentType` enum + `select_agent` byllm router,
  session persistence, `interact` / `get_session` walkers, CLI REPL.
- `rag_engine.jac` — FAISS vector store over `docs/` (local HuggingFace
  embeddings, chunking, metadata-enriched chunks) with CrossEncoder reranking.
- `config/faiss_reranking.json` — model + RAG parameters (chunking, k,
  reranker); loaded at startup, falls back to built-in defaults.

## Setup

```bash
export OPENAI_API_KEY=sk-...   # or put it in .env
jac install                    # installs Python deps + byllm capability
```

First run builds the FAISS index from `docs/` (one-time embedding cost);
subsequent runs load it from `faiss_index/`. The CrossEncoder reranker model
downloads from HuggingFace on first use; if unavailable, reranking is
disabled automatically.

## Run

Terminal chat:

```bash
jac run main.jac
```

As a service (REST):

```bash
jac start main.jac
# POST /walker/interact     {"message": "...", "session_id": "s1"}
# POST /walker/get_session  {"session_id": "s1"}
```

`MODEL=gpt-4o jac run main.jac` overrides the LLM from the environment.
