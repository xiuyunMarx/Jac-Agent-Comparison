# Coding agent — LangGraph

The LangGraph half of a two-framework comparison. The other half is
[`../byLLM`](../byLLM), the same agent written in Jac with byLLM. Both sides are
benchmarked against the SWE-bench harness vendored at `../SWE-bench`, so they
must present an **identical action space**: the same ten tools, the same string
contracts, the same path confinement, the same command allowlist, the same
limits. The four tool modules are therefore a near-literal port, and the
interesting difference between the two implementations is confined to how the
phase graph and the ReAct loop are expressed.

## Running it

```bash
pip install -e '.[dev]'
export OPENAI_API_KEY=...            # read natively by langchain-openai
export CODEAGENT_MODEL=gpt-4o   # optional; same default as byLLM/jac.toml

python orchestrator.py "add a docstring to tools/plan.py" .
python -m pytest -q                  # 94 tests, no network, no API key
```

The last line prints the same telemetry shape as the byLLM CLI:

```
[4 phases, 12 tool calls, 9 llm calls, 18422+733 tokens]
```

To see the compiled phase graph:

```bash
python -c "import orchestrator; print(orchestrator.build_app().get_graph().draw_mermaid())"
```

## The graph

Five phases. Every phase can read the repository and update the plan; only
Editing can write; only Editing and Verifying can execute.

```
        ┌──────────┐
        │ Planning │──────────────┐          set_plan, read tools
        └────┬─────┘              │
             ▼                    │
        ┌───────────┐             │          read tools
        │ Exploring │─────────────┤
        └────┬──────┘             │
             ▼                    ▼
        ┌─────────┐          ┌──────────┐
        │ Editing │          │ Finished │    write + run tools
        └────┬────┘          └──────────┘
             ▼                    ▲
        ┌───────────┐             │
        │ Verifying │─────────────┘          run tools
        └────┬──────┘
             │  objective not met
             └──────────► Editing            the repair loop
```

`FLOW` in `orchestrator.py` declares that topology once, and `build_app` derives
each `add_conditional_edges` path map from it — so a router can never send the
run somewhere the declared topology does not allow, and
`test_the_compiled_graph_wires_exactly_the_declared_topology` checks the
compiled graph against it.

## Layout

| file | what it holds |
| --- | --- |
| `orchestrator.py` | model seam, phase table, `FLOW`, graph build, `solve()`, CLI |
| `phase_agent.py` | the per-phase ReAct loop, progress guard, serializing tool node |
| `telemetry.py` | token accounting |
| `tools/common.py` | limits, tool-call log, path confinement, clipping, scrubbing |
| `tools/explore.py` | `read_file`, `ls_repo`, `find_files`, `grep` |
| `tools/edit.py` | `write_file`, `replace_in_file` |
| `tools/plan.py` | `set_plan`, `update_task`, `show_plan` |
| `tools/verify.py` | `run_command`, local and docker backends |

The directory is `tools/` rather than byLLM's `nodes/` because in a LangGraph
codebase "node" means a graph node, and these modules are not graph nodes. The
module basenames match so the two trees diff cleanly.

## byLLM → LangGraph

| byLLM (Jac) | LangGraph (Python) |
| --- | --- |
| `node Tool`, `edge Exposes` | frozen `ToolSpec(tool, category)` in `Phase.exposes` |
| `[self ->:Exposes:->]` | `Phase.tools()` / `Phase.tool_names()` |
| `edge Flow: Phase --> Phase` | `FLOW: dict[str, tuple[str, ...]]` |
| `node Planning(Phase)` … | entries in `build_pipeline()`, added with `graph.add_node` |
| `walker CodeAgent`, `visit [->:Flow:->][?:X]` | `AgentState` + `add_conditional_edges` routers |
| `Phase.ctx` | `AgentState["phase_msgs"][title]` |
| `def run_phase(...) by agent_model(tools=…)` | the compiled subgraph in `phase_agent.py` |
| `sem` strings | `description=` on `StructuredTool`, `Field(description=…)` on args schemas |
| `TaskStatus` enum at the LLM boundary | `Literal["todo","doing","done","blocked"]` |
| `mark_serialize` | `SerialToolNode` |
| litellm `success_callback` + `settle()` | `AIMessage.usage_metadata`, read synchronously |
| `MockLLM` / `MockToolCall` | `FakeToolCallingModel` in `tests/conftest.py` |

### What LangGraph needed that byLLM did not

**The ReAct loop is hand-built.** byLLM gets the whole thing from one clause:

```jac
def run_phase(directive: str) -> str by agent_model(
    tools=rt.phase_tools, max_react_iterations=25,
    on_iteration=progress_guard, max_tool_result_length=4000
);
```

LangGraph 0.3.5 has `create_react_agent`, but it cannot express two of those
four behaviours: it has a `pre_model_hook` and no `post_model_hook`, and no hook
can end the loop, so the progress guard could only be advisory; and its
iteration cap is a `recursion_limit` that raises `GraphRecursionError` rather
than letting the model summarise. A phase that ends with no text contributes
nothing to the ledger and the next phase then works blind, so `phase_agent.py`
builds the loop out of primitives instead — `model → tools → guard → model`,
with both brakes routing to a `summarize` node.

**Parallel tool calls had to be re-serialized.** `langgraph.prebuilt.ToolNode`
dispatches a batch through `executor.map`, so a turn asking for `write_file` and
`run_command` together runs them concurrently and the run stops being
reproducible — fatal for an A/B benchmark. `SerialToolNode` feeds any batch
naming a mutating tool through ToolNode one call at a time; read-only batches
keep the parallelism.

### What LangGraph made simpler

**Token accounting.** byLLM hooks `litellm.success_callback` and then polls in
`TokenUsage.settle()` until the records stop moving, because litellm runs those
callbacks on a background pool. LangChain returns usage on the response object
synchronously, so both the registration and the settle loop are gone.

**No smuggled state.** byLLM's `AgentRuntime` carries `phase_tools` and
`phase_ctx` because byLLM serializes a `by` function's parameters into the
prompt, so passing the tool list or the conversation as arguments would dump
them into every request as object reprs. LangGraph passes both explicitly, so
those two fields do not exist here.

## Tool-side guards

Three behaviours in `tools/` exist to stop a phase burning its budget on a loop
rather than on the task, and are ported from the byLLM side unchanged:

- **`read_file` served-window memory.** A file small enough to fit comes back
  whole whatever window was asked for, and a window already served *in this
  phase* is answered with the next move — the exact `start_line` to resume at,
  or a pointer to `grep` — instead of the same bytes. The memory key carries the
  file's size and mtime, so a write expires it and Editing's read-back after an
  edit is served normally. Scope is stamped per phase by `make_phase_node`,
  because phases keep separate conversations. Oversized files page forward on
  whole lines, so the continuation hint names a line that `start_line` can
  actually resume from and edit anchors are never silently clipped.
- **Post-write syntax check.** `write_file` and `replace_in_file` parse what
  they just wrote — Python via `compile`, Jac via the jaclang parser when it is
  installed — and append a `WARNING` naming the line when it no longer parses.
  The write still goes through: reverting behind the model's back would leave it
  reasoning about a file that is not the one on disk. Formats with no parser are
  left alone.
- **Terse `pytest` defaults.** `-q --tb=line` are supplied only when the command
  did not already decide the matter, so `pytest --tb=long` keeps its frames. The
  echoed `$ …` header shows the argv that actually ran, injected flags included,
  so a one-line traceback reads as a default the model may override rather than
  as pytest having lost the frames.

## Action-space parity

Both implementations were run against identical inputs and their outputs
compared byte for byte:

| tool group | cases | identical |
| --- | --- | --- |
| `run_command` refusals | 36 | 30 |
| docker allowlist + `_wrap_docker` argv | 2 | 2 |
| `read_file` / `ls_repo` / `find_files` / `grep` | 18 | 17 |
| `write_file` / `replace_in_file` (message *and* resulting bytes on disk) | 18 | 18 |
| `set_plan` / `update_task` / `show_plan` | 4 | 4 |

The seven differences are all one byLLM-side defect, and none of them changes
what the agent is allowed to do — only the wording the model reads back.

**Jac f-strings drop quotes around the second of two interpolations.** A Jac
f-string containing two quoted interpolations loses the inner pair:

```jac
with entry {
    bare = "--pdb"; prog = "pytest";
    print(f"the flag '{bare}' is not allowed for '{prog}'.");
}
# prints: the flag '--pdb is not allowed for pytest'.
```

One quoted interpolation is fine; the delimiter can be either quote character
and the other one is affected. Two byLLM sites hit it:

- `nodes/verify.jac:408` — the denied-flag refusal, which is why 6 of the 36
  `run_command` cases differ.
- `nodes/explore.jac:221` — grep's no-match sentence, which byLLM renders as
  `No matches for 'x' in '. (file_glob=*', searched 2 files).`

This project renders what the Jac source intends. Fixing the Jac side (or the
Jac compiler) is a separate call; the byLLM tests do not currently assert the
affected substrings, so they pass either way.

## Tests

118 pytest tests, a 1:1 port of the 6 byLLM test files plus coverage for the
machinery byLLM does not need:

| file | covers |
| --- | --- |
| `test_common.py` | path confinement, symlink and sibling-prefix escapes, clipping, scrubbing |
| `test_explore.py` | read windows, whole-line paging, served-window memory and its per-phase scope, sorted listings, glob matching, grep formatting and errors |
| `test_edit.py` | anchored edits, whitespace-drift diagnostics, CRLF round-trip, write refusals, the post-write syntax check |
| `test_plan.py` | checklist rendering, status transitions, the JSON-Schema enum |
| `test_verify.py` | the allowlist, denied and default flags, shell metacharacters, the echoed argv, the timeout kill, both backends |
| `test_orchestrator.py` | the capability boundary, the declared vs compiled topology, both loop brakes, tool serialization, result clipping, full traversals |

`FakeToolCallingModel` replaces byLLM's `MockLLM`; `GenericFakeChatModel` from
langchain-core cannot be used, since it inherits `BaseChatModel.bind_tools`,
which raises `NotImplementedError` and so can never drive a tool-calling loop.
