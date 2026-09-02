# Coding agent — no framework, just the OpenAI SDK

One of three implementations of the same agent. The original is
[`../Jac`](../Jac) (Jac + byLLM); [`../langgraph`](../langgraph) is the
LangGraph port; this one is the no-framework baseline. All three present the
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
```

## Layout — one file per Jac file

| this | `../Jac` | what it holds |
| --- | --- | --- |
| `nodes.py` | `nodes.jac` | model seam, shared run state (`history`, `tool_log`, `TOOL_BUDGET`), the `Repo` toolbox and its `sem` strings, the phase nodes and the `PhaseCapability` edge |
| `phase_agent.py` | *(the `by llm(...)` clause)* | the ReAct loop byLLM runs for `work()`: its system message, call-frame prompt, `finish_tool`, forced final answer, write-back into `history`; and `select_edge`, its visit router |
| `main.py` | `main.jac` | `CodeAgent`, the walk, the Verify guards and routing, `solve()` |
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

Plan, Explore and Edit each take their one forward edge. Verify runs the
deterministic guards first (repair budget, tool budget, "did a write land",
"did anything run since"), and only then asks the model to pick an edge; the
relocate edge is offered once, after a first repair has also failed.

## What byLLM was doing

The `by llm(tools=..., conversation=history, max_react_iterations=N,
on_iteration=guard)` clause on each `work()` is the whole of `phase_agent.py`:
byLLM's own system message and call-frame prompt, the `finish_tool` that ends a
phase, sequential tool dispatch, the guard before every round, the
"provide only your final answer" nudge on abort, and the write-back that keeps
one conversation running through every phase. `visit [...] by llm(select=1,
intent=..., incl_info=...)` is `select_edge`: the candidates rendered as
byLLM's router renders them, answered under a JSON schema over the handles.

## SWE-bench

Wired into the shared bridge as `--framework openai`; the shim imports
`solve` / `active_model_name` / `DEFAULT_MODEL` from `orchestrator`, which
returns the same `RunResult` fields as the Jac side.
