# Coding agent — no framework, just the OpenAI SDK

The third implementation of one agent. The others are [`../byLLM`](../byLLM)
(Jac + byLLM) and [`../langgraph`](../langgraph) (Python + LangGraph). All three
are benchmarked against the SWE-bench harness vendored at `../SWE-bench`, so
they must present an **identical action space**: the same ten tools, the same
string contracts, the same path confinement, the same command allowlist, the
same limits. The tool modules are therefore a near-literal port, and the whole
of the interesting difference is in how the phase graph, the ReAct loop and the
accounting are expressed.

This side is the baseline the other two are measured against: what a framework
is actually buying over `while True:` and `client.chat.completions.create`.

**Fidelity target is `byLLM`, not `langgraph`.** Where the two existing sides
have drifted apart, this one follows the Jac original — see
[Where langgraph and byLLM disagree](#where-langgraph-and-byllm-disagree).

## Running it

```bash
pip install -e .
export OPENAI_API_KEY=...             # read natively by the openai SDK
export CODEAGENT_MODEL=gpt-4o         # optional; same default as the other two

python orchestrator.py --task "add a docstring to tools/plan.py" --repo-path .
```

The CLI flags are byLLM's (`--task`, `--repo-path`). stdout carries the answer
and nothing else, because the eval harness reads it; the telemetry line and the
call-log note go to stderr:

```
[4 phases, 12 tool calls, 9 llm calls, 18422+733 tokens]
[llm-log] 9 call(s) written to llm_calls/
```

`llm_calls/` is the per-call transcript — one file per round trip, plus an
`index.txt` — ported from byLLM's `logger/log_LLM_history.jac`. Point it
somewhere else with `$CODEAGENT_LLM_LOG`.

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

`FLOW` in `orchestrator.py` declares that topology once, with the reason for
every edge. A phase with one outgoing edge takes it; a phase with two asks the
model, and the model may only answer with a target `FLOW` declares — so a router
can never send the run somewhere the topology does not allow.

## Layout

| file | what it holds | lines |
| --- | --- | ---: |
| `orchestrator.py` | client seam, phase table, `FLOW`, the edge router, the walk, `solve()`, CLI | 572 |
| `phase_agent.py` | the per-phase ReAct loop, both progress brakes, tool dispatch | 310 |
| `llm.py` | one `chat.completions.create`, recorded | 183 |
| `telemetry.py` | the `LLMCall` record and token totals | 172 |
| `llm_log.py` | the `llm_calls/call-NNN.txt` writer | 225 |
| `tools/spec.py` | `Tool` + JSON-Schema helpers — what `StructuredTool` was doing | 182 |
| `tools/common.py` | limits, tool-call log, path confinement, clipping, scrubbing | 158 |
| `tools/explore.py` | `read_file`, `ls_repo`, `find_files`, `grep` | 462 |
| `tools/edit.py` | `write_file`, `replace_in_file` | 335 |
| `tools/plan.py` | `set_plan`, `update_task`, `show_plan` | 158 |
| `tools/verify.py` | `run_command`, local / docker / udocker backends | 663 |

The directory is `tools/` to match `../langgraph/tools/`; byLLM calls the same
four modules `nodes/`. The basenames match on all three sides so the trees diff
cleanly.

## byLLM → LangGraph → no framework

| byLLM (Jac) | LangGraph | this |
| --- | --- | --- |
| `node Tool`, `edge Exposes` | frozen `ToolSpec` in `Phase.exposes` | same |
| `edge Flow: Phase --> Phase { has reason }` | `FLOW: dict[str, tuple[str, ...]]` | `FLOW: dict[str, tuple[FlowEdge, ...]]`, reason kept |
| `node Planning(Phase)` … | `graph.add_node` per phase | entries in `build_pipeline()` |
| `walker CodeAgent`, `visit [->:Flow:->]` | `AgentState` + `add_conditional_edges` | `AgentRun` + `walk()`'s `while` loop |
| `Phase._ctx` | `AgentState["phase_msgs"][title]` | a fresh `[system]` list per visit |
| `def run_phase(...) by agent_model(tools=…)` | a compiled sub-`StateGraph` | `phase_agent.run_phase`, a `while` loop |
| `sem` strings | `description=` + `Field(description=…)` | `Tool.description` + `prop(..., "…")` |
| `TaskStatus` enum at the LLM boundary | `Literal["todo",…]` | `enum: [...]` in the JSON Schema |
| `mark_serialize` | `SerialToolNode` | *(nothing — a `for` loop is serial)* |
| `visit [edge …] by agent_model(select=1, intent=…)` | fixed edges + a `bool` classifier | `select_edge`, strict `json_schema` over the candidate titles |
| litellm `success_callback` + `settle()` | `usage_metadata`, read synchronously | `response.usage`, read at the call site |
| `logger/log_LLM_history.jac` | *(absent)* | `llm_log.py` |

### What the framework was doing, and what replaced it

**The ReAct loop.** byLLM gets the loop, the tool schemas, the iteration cap, the
per-iteration hook and the result clipping from one `by` clause. LangGraph
builds it out of graph primitives (`model → tools → guard → model`) because
`create_react_agent` cannot express two of those four behaviours. Here it is
~90 lines of `while True:` in `phase_agent.py`. That is the single largest thing
a framework was providing, and it is also the part with the most behaviour to
get wrong — the cap, the brakes and the abort path all have to end the phase
*with a summary*, because a phase that ends silently contributes nothing to the
ledger and the next phase then works blind.

**The tool schema.** `StructuredTool.from_function` + a pydantic argument model
becomes a `Tool` dataclass and an explicit JSON Schema. Argument validation and
the lax-mode coercions pydantic was doing (`"12"` → `12`, and no more than that)
are ~40 lines in `tools/spec.py`. The schemas this produces are byte-identical
to LangGraph's, `required` list included:

```bash
python -c "
import json,sys; sys.path.insert(0,'.')
from tools import ExploreCodeBase, EditCode, PlanTasks, VerifyCode
print(json.dumps({t.name: t.openai_spec() for h in
  (ExploreCodeBase('.'), EditCode('.'), PlanTasks(), VerifyCode('.'))
  for t in h.as_tools()}, indent=2, sort_keys=True))" > /tmp/mine.json
# then the same for ../langgraph via convert_to_openai_tool, and diff.
```

**Nothing at all, for tool serialization.** A turn asking for `write_file` and
`run_command` together must not run them concurrently, or the run stops being
reproducible — fatal for an A/B benchmark. byLLM needs `mark_serialize`;
LangGraph needs a `SerialToolNode` subclass, because `langgraph.prebuilt.ToolNode`
dispatches a batch through `executor.map`. A `for` loop over the batch is
already serial, so this side has no code for it.

**Nothing at all, for the settle loop.** byLLM hooks `litellm.success_callback`
and then polls in `TokenUsage.settle()` until the records stop moving, because
litellm runs those callbacks on a background pool and they lag the last call —
and the miss is silent, making a run look one call cheaper than it was. Usage
arrives on the response object here, so the record is complete before `complete()`
returns.

**Roughly break-even on lines.** Orchestration only, excluding `tools/`:

| | byLLM | LangGraph | this |
| --- | ---: | ---: | ---: |
| phase graph + loop | 461 | 732 | 882 |
| token accounting | 209 | 60 | 172 |
| per-call transcript log | 242 | — | 225 |

The comment density is the same on all three sides, so these compare like for
like. Note where the cost lands: the no-framework side pays for the loop and
the router, and gets the accounting back cheaply.

## Where LangGraph and byLLM disagree

Four places, and this side follows byLLM at each:

1. **The router.** byLLM asks the model which edge to take at Planning,
   Exploring *and* Verifying. LangGraph hard-wired the first two and replaced
   the third with a boolean `objective_met` classifier, which makes the
   read-only shortcut to `Finished` unreachable. `select_edge` restores the
   three-way choice, and the `intent` strings are byLLM's verbatim.
2. **The second progress brake.** byLLM aborts a phase that reaches for the same
   tool six times with nothing in between, which the identical-call test cannot
   see: `read_file` walking a file one line at a time issues a different call
   every time. LangGraph has only the identical-call brake.
3. **The tool-result budget.** byLLM ties `max_tool_result_length` to the tools'
   own `MAX_TOOL_CHARS` (20000), deliberately not below it, because a second
   truncation would cut mid-line and destroy the continuation hint the tool put
   there. LangGraph clips to 4000.
4. **The per-phase conversation.** byLLM rebuilds it from the system message on
   every visit; only the ledger crosses. LangGraph keeps the previous lap's tool
   traffic, which is what put four full pytest runs into Editing's fourth lap —
   22k of its 28k prompt tokens, all describing a file it had since edited three
   more times.

## Tool-side guards

Three behaviours in `tools/` exist to stop a phase burning its budget on a loop
rather than on the task, ported unchanged from the other two sides:

- **`read_file` served-window memory.** A file small enough to fit comes back
  whole whatever window was asked for, and a window already served *in this
  phase* is answered with the next move — the exact `start_line` to resume at,
  or a pointer to `grep` — instead of the same bytes. The memory key carries the
  file's size and mtime, so a write expires it and Editing's read-back after an
  edit is served normally. Scope is stamped per phase by `walk()`.
- **Post-write syntax check.** `write_file` and `replace_in_file` parse what
  they just wrote — Python via `compile`, Jac via the jaclang parser when it is
  installed — and append a `WARNING` naming the line when it no longer parses.
  The write still goes through: reverting behind the model's back would leave it
  reasoning about a file that is not the one on disk.
- **Terse `pytest` defaults.** `-q --tb=line` are supplied only when the command
  did not already decide the matter. The echoed `$ …` header shows the argv that
  actually ran, injected flags included.

## SWE-bench

Wired into the shared bridge as `--framework openai`:

```bash
python ../swebench_bridge/run_agent.py --framework openai \
    --instances astropy__astropy-12907 --run-id smoke --model gpt-4o
```

The bridge hands every framework the same workspace, container, objective text
and patch extraction; the only framework-specific thing is which shim gets
spawned. This side reuses `swe_entry.py` unchanged — it resolves the agent from
`$CODEAGENT_HOME` and imports `solve` / `active_model_name` / `DEFAULT_MODEL`
from `orchestrator`, which this module exports with byLLM's field set.
