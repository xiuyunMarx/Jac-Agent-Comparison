# Coding agent — LangGraph

One of three implementations of the same agent. The original is
[`../Jac`](../Jac) (Jac + byLLM); [`../openai_sdk`](../openai_sdk) is the
no-framework baseline; this one is the LangGraph port. All three present the
same eight tools with the same string contracts, the same phase goals, the same
shared conversation across phases, the same guards and the same budgets, so a
difference in score is a difference between frameworks and not between agents.

## Running it

```bash
pip install -e .
export OLLAMA_API_KEY=...             # ollama cloud; or OPENAI_API_KEY for any /v1 server
export CODEAGENT_MODEL=openai/glm-5.2 # same default as ../Jac; the prefix is stripped on the wire

python main.py "issue text" /path/to/repo
python tests/tool_checks.py            # tool layer, no LLM
python tests/smoke.py                  # a toy bug, end to end
python -c "import main; print(main.build_app().get_graph().draw_mermaid())"
```

## Layout — one file per Jac file

| this | `../Jac` | what it holds |
| --- | --- | --- |
| `nodes.py` | `nodes.jac` | model seam, shared run state (`history`, `tool_log`, `TOOL_BUDGET`), the `Repo` toolbox and its `sem` strings as `StructuredTool`s, the phase nodes and the `PhaseCapability` edge |
| `phase_agent.py` | *(the `by llm(...)` clause)* | the ReAct loop byLLM runs for `work()` as a compiled `model -> tools -> summarize` StateGraph, with byLLM's system message, call-frame prompt, `finish_tool`, forced final answer and write-back into `history`; and `select_edge`, its visit router |
| `main.py` | `main.jac` | `AgentState`, the phase StateGraph, the Verify guards and routing, `solve()` |
| `orchestrator.py` | `orchestrator.jac` | the swebench_bridge adapter: `RunResult`, token totals, `solve()` |
| `tests/` | `tests/` | `tool_checks` (no LLM) and `smoke` (needs a key) |

## The graph

```
Root -> Plan -> Explore -> Edit -> Verify -> Finish
                  ^          ^        |
                  |          |        +-- repair   -> Edit
                  |          +----------- (forward)
                  +---------------------- relocate -> Explore
                                          finish   -> Finish
```

Plan, Explore and Edit are `add_edge`s. Verify's decision is made inside the
Verify node, because the Jac ability mutates `repairs`, `relocates` and the
ledger while deciding and a LangGraph router must not; the conditional edge
only reads the choice back. The deterministic guards (repair budget, tool
budget, "did a write land", "did anything run since") come before the model,
and the relocate edge is offered once, after a first repair has also failed.

## What byLLM was doing

`by llm(tools=..., conversation=history, max_react_iterations=N,
on_iteration=guard)` on each `work()` is the whole of `phase_agent.py`: byLLM's
own system message and call-frame prompt, the `finish_tool` that ends a phase,
sequential tool dispatch (one `ToolNode` call per tool call), the guard before
every round, the "provide only your final answer" nudge on abort, and the
write-back that keeps one conversation running through every phase. Because
`add_messages` copies messages, scaffolding is marked in `additional_kwargs`
rather than tracked by identity. `visit [...] by llm(select=1, intent=...,
incl_info=...)` is `select_edge`: `ChatOpenAI.bind(response_format=...)` with
byLLM's own JSON schema for a `list` (`schema_object_wrapper`), byLLM's schema
hint at the end of the prompt, and its one correction retry when the first
answer is not JSON. GLM over ollama's `/v1` answers that first call in prose
every time, so a route costs two calls on every arm alike. Tools are bound as
the OpenAI-format specs (`nodes.spec`), not as the `StructuredTool`s, because
LangChain's own conversion drops the `additionalProperties: false` byLLM sends.

Set `CODEAGENT_TRACE=<file>` and every model call is appended there as one
JSON line: the request as LangChain put it on the wire (in full when the call
opens a phase or routes, else the newest message), the usage and the reply.
The bridge sets it to `logs/<instance>/llm_trace.jsonl`, and
`swebench_bridge/trace_diff.py` diffs those files across arms; with the same
model, the same instance and the same tool results, the traces of the three
arms are identical byte for byte, so the numbers differ only where the model
chose differently.

## SWE-bench

Wired into the shared bridge as `--framework langgraph`; the shim imports
`solve` / `active_model_name` / `DEFAULT_MODEL` from `orchestrator`, which
returns the same `RunResult` fields as the Jac side.
