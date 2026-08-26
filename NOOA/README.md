# Coding agent — NVIDIA NOOA, one object, no phase graph

The fourth implementation of one agent. The others are
[`../CodeAgent/byLLM`](../CodeAgent/byLLM) (Jac + byLLM),
[`../CodeAgent/langgraph`](../CodeAgent/langgraph) (Python + LangGraph) and
[`../CodeAgent/openai_sdk`](../CodeAgent/openai_sdk) (Python, no framework). All
four are benchmarked against the SWE-bench harness vendored at
`../CodeAgent/SWE-bench`, so they must present an **identical action space**: the
same ten tools, the same string contracts, the same path confinement, the same
command allowlist, the same limits. `tools/` is therefore a near-literal port,
and the whole of the interesting difference is what happens above it.

On the other three sides, what happens above it is a **phase graph**: five
phases, each granted a subset of the tools, wired by edges, walked by something
that asks the model at every fork which edge to take. This side does not have
one, and that is the point of adding it.

[NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents) — NVIDIA's
Object-Oriented Agents — is model-agnostic and built on one idea: an agent is a
Python object, its capabilities are its methods, and the model acts by **writing
Python** in a session where `self` is in scope. A method with a real body is
deterministic code; a method with a `...` body is one the LLM implements, with
its docstring as the prompt and its annotations as the contract. So there is no
tool schema to register, no ReAct loop to write, and no reason to cut the task
into phases before the model sees it: "grep, read the two files it points at,
edit one, run the tests, read the failure, edit again" is a program, and the
model can just write it.

What a benchmark still needs is guarantees, and those do not come from a graph
either. Here they are three checks that ask the model nothing — see
[The three guards](#the-three-guards).

## Running it

```bash
pip install nooa                      # the runtime; 0.0.9 is what this was written against
export OPENAI_API_KEY=...             # or ANTHROPIC_API_KEY, or a local endpoint
export CODEAGENT_MODEL=gpt-5          # optional; same default as the other three

python orchestrator.py --task "make greet return world" --repo-path .
```

The CLI flags are byLLM's (`--task`, `--repo-path`). stdout carries the answer
and nothing else, because the eval harness reads it; the telemetry line goes to
stderr:

```
[2 steps, 11 tool calls, 23 llm calls, 41022+3117 tokens]
```

The model is named rather than configured, litellm-style, so anything litellm
routes works — `gpt-5`, `claude-sonnet-5`, `ollama_chat/qwen3` with
`CODEAGENT_API_BASE` for a local server. **A tool-calling model is required**:
CodeAct drives the session through an `execute_python` tool, so a model without
tool support (`llama3` in ollama, for instance) fails at the first call.

There is no per-call transcript log here — byLLM's `logger/log_LLM_history.jac`
and openai_sdk's `llm_log.py` — because the runtime has one. `pip install
nooa-cli && nooa start-dev` serves a trace viewer on :5001, and every LLM call,
cell and method invocation lands in it.

## The shape

One object. `CodeAgent` in `agent.py` holds the ten tools as methods, the plan as
state, and two methods it does not implement:

```python
class CodeAgent(Agent):
    """You are a coding agent working in one repository, on one objective. ..."""

    def grep(self, pattern: str, path: str = ".", file_glob: str = "*") -> str:
        """Search file contents for a regular expression. ..."""
        return self.repo.grep(pattern, path, file_glob)

    @strategy(CodeActStrategy(config=WORK))
    async def resolve(self, objective: str) -> Report:
        """Deliver this objective in the repository, then report what you did. ..."""
        ...
```

and one it does, which is the whole of the orchestration:

```python
    @hidden
    async def run(self, objective: str, max_steps: int = 10) -> Report:
        self.steps += 1
        report = await self.resolve(objective)
        checked = self._gate(report)
        while not (report.resolved and checked.passed) and self.steps < max_steps:
            self.steps += 1
            report = await self.repair(objective, report, checked.rendered())
            checked = self._gate(report)
        ...
```

`@hidden` keeps `run` out of what the model is shown, so a generated cell cannot
call the workflow back on itself. That is NOOA's documented pattern for an entry
point: model judgement in methods with contracts, control flow in ordinary
Python, chained by something hidden.

A step is one generated-method call rather than one phase visit, so
`--max-steps 10` buys `resolve` plus nine repairs, each of up to 25 cells — the
same ceiling the other three put on one phase, against the same numbers.

## Layout

| file | what it holds | lines |
| --- | --- | ---: |
| `agent.py` | the agent: ten capabilities, `resolve`, `repair`, the guards, `run` | 621 |
| `orchestrator.py` | `RunResult`, `solve`, the CLI — the seam the bridge imports | 165 |
| `llm.py` | which model, and the one client it is called on | 119 |
| `telemetry.py` | the meter: token usage, read off every response | 153 |
| `tools/common.py` | limits, tool-call log, path confinement, clipping, scrubbing | 164 |
| `tools/explore.py` | `read_file`, `ls_repo`, `find_files`, `grep` | 351 |
| `tools/edit.py` | `write_file`, `replace_in_file` | 276 |
| `tools/plan.py` | `set_plan`, `update_task`, `show_plan` | 87 |
| `tools/verify.py` | `run_command`, local / docker / udocker backends | 629 |

There is no `tools/spec.py`, no `phase_agent.py`, and no phase module. The
basenames under `tools/` match the other three sides so the trees diff cleanly;
byLLM calls the same four modules `nodes/`.

## byLLM → LangGraph → no framework → this

| byLLM (Jac) | LangGraph | openai_sdk | this |
| --- | --- | --- | --- |
| `node Tool`, `edge Exposes` | frozen `ToolSpec` in `Phase.exposes` | same | *(nothing — a method is the tool)* |
| `sem` strings | `description=` + `Field(description=…)` | `Tool.description` + explicit JSON Schema | the method's own docstring and annotations |
| `TaskStatus` enum at the LLM boundary | `Literal["todo",…]` | `enum: [...]` in the schema | `Literal["todo",…]` — the annotation *is* the boundary |
| `edge Flow: Phase --> Phase { has reason }` | `FLOW: dict[str, tuple[str, ...]]` | `FLOW: dict[str, tuple[FlowEdge, ...]]` | *(nothing — the model writes the sequence)* |
| `node Planning(Phase)` … | `graph.add_node` per phase | entries in `build_pipeline()` | *(nothing)* |
| `walker CodeAgent`, `visit [->:Flow:->]` | `AgentState` + `add_conditional_edges` | `AgentRun` + `walk()`'s `while` | `run()`: one `await`, then a `while` over the gate |
| `visit [edge …] by agent_model(select=1, intent=…)` | a `bool` classifier | `select_edge` + strict `json_schema` | *(nothing — `Report.resolved` is a field of the result)* |
| `Phase._ctx` | `AgentState["phase_msgs"][title]` | a fresh `[system]` list per visit | one session; NOOA carries the event history across method calls |
| `def run_phase(...) by agent_model(tools=…)` | a compiled sub-`StateGraph` | `phase_agent.run_phase`, a `while` loop | `@strategy(CodeActStrategy(config=WORK))` |
| `mark_serialize` | `SerialToolNode` | *(a `for` loop is serial)* | *(a cell is a program; it runs its calls in order)* |
| `on_iteration` progress brakes | the identical-call brake only | both brakes | *(deliberately absent — see below)* |
| litellm `success_callback` + `settle()` | `usage_metadata` | `response.usage` at the call site | `telemetry.attach`, wrapping the client's one method |
| `logger/log_LLM_history.jac` | *(absent)* | `llm_log.py` | the runtime's own tracing (`nooa start-dev`) |

### What the framework was doing, and what replaced it

**The ReAct loop.** byLLM gets it from one `by` clause; LangGraph builds it out
of graph primitives; openai_sdk writes ~90 lines of `while True:`. Here it is
`@strategy(CodeActStrategy(config=CodeActConfig(max_iterations=25)))` — the
runtime owns the loop, the cap, the tool dispatch and the result rendering. That
is the largest single thing a framework provides in this comparison, and it is
the one NOOA provides most completely.

**The tool schema — deleted, not ported.** The other three sides all end up
producing the same JSON Schema by three different routes (a `sem` string, a
pydantic argument model, 182 hand-written lines in `tools/spec.py`). There is no
schema here at all: the model calls the method, so the docstring is the
description and the annotation is the type. The 177 lines of tool prose that
were `*_DOC` and `*_PARAMS` constants are now docstrings on the methods, and
`tools/` came out 275 lines shorter for the same ten tools.

**The router — deleted.** "Editing or Finished?" is not a question anyone has to
ask when the model is writing the control flow: it decides by writing
`await self.repair(...)` or by returning. And "am I done?" is not a separate LLM
call either — it is `Report.resolved`, a field of the value the method already
returns. Every arm but this one spends a second class of LLM call on routing.

**The per-phase conversation — deleted.** openai_sdk rebuilds a phase's
messages from its system prompt on every visit, and byLLM does the same, because
keeping the previous lap's tool traffic put four full pytest runs into one
phase's fourth lap (22k of its 28k prompt tokens). That whole problem is a
consequence of the phase graph. Here one session does the task, and NOOA's event
history is available across generated-method calls by default, so `repair` sees
what `resolve` did without anything having to be summarised and handed forward.

**The accounting — cheap.** byLLM registers a litellm `success_callback` and then
polls in `settle()` until the records stop moving, because usage arrives on a
background pool and a missed final call makes a run look cheaper than it was.
Nothing in this arm's code ever sees a response — the runtime owns the loop — so
`telemetry.attach` wraps the client's one entry point and records on the way
back, on the calling task, before the value is handed on. 153 lines, no settle
loop, and `cached_tokens` populated.

## The three guards

None of them asks the model to assert anything. The first two are NOOA
postconditions on `resolve` and `repair`, so a failure comes back as
`InvariantError` and is fed into the session as a correctable error — the model
fixes the omission with the whole conversation still in front of it, where the
other three sides can only route the walker back to a phase whose conversation
starts empty.

| guard | what it asserts | why |
| --- | --- | --- |
| `must_have_written` | a report claiming a fix has a write in the tool log | a phase reporting "Summary of changes made: modified …/fields/__init__.py" having called only `grep` and `read_file` is describing the edit it decided on and never issued. Only the log knows. |
| `must_have_verified` | something was run **after** the last write | a repository's existing suite passes on an unfixed tree, because the test that would catch the bug is held back. A green run from before the edit is evidence of nothing. |
| `check` (in `run`) | the command the report offers as proof is **re-run**, and its exit code decides | *no other arm can do this.* |

The third is worth spelling out. `Report` carries a `verify_command`, and `run`
executes it again through the agent's own `run_command` — so the gate cannot run
anything the model could not have run itself, and the re-run appears in the tool
log like every other call. An empty command fails. A refused command fails. Any
non-zero exit fails. And if the budget runs out with the gate still failing, the
final report is **downgraded to `resolved: False`** before it is recorded,
because the answer the harness stores must not claim more than was demonstrated.

A phase in a graph reports and the walker moves on; there is nothing left
afterwards that could re-run what it claimed. This is the one place where not
having a graph buys a guarantee rather than costing one.

## Where this arm deliberately differs

Four places, each a consequence of code-as-action rather than a preference:

1. **No per-call progress brakes.** byLLM aborts a phase that issues the
   identical tool call three times running, or reaches for the same tool six
   times with nothing in between; openai_sdk ports both. Both brakes assume one
   tool call per model turn, which is exactly the assumption a cell removes — a
   loop calling `grep` six times is one turn here, and is often the right thing
   to write. What survives is the guard that was never a loop heuristic:
   `read_file`'s served-window memory, which answers a repeat with the next move
   instead of the same bytes, and expires when the file is written to.
2. **A step is a generated-method call**, not a phase visit. `runs.jsonl` will
   show single-digit `steps` against the others' double digits for comparable
   work; `llm_calls` and the token columns are the ones to compare.
3. **One truncation, not two.** The tools clip themselves at 20,000 characters,
   in terms the model can act on — `read_file` stops on a line boundary and
   names the `start_line` that resumes it. NOOA's per-cell stdout budget is
   50,000 and head/tail, so the tools' own truncation is the one the model sees,
   which is the same choice byLLM made and openai_sdk kept.
4. **`nooa.tools.shell_tools.ShellTools` and `TodoManager` are not used**,
   though they ship with the runtime and would be the idiomatic choice for a
   fresh agent. They are not the same action space, and a difference in the
   score has to be a difference between the frameworks.

## What confines this agent

Less than the other three arms, and the difference is worth being precise about
rather than glossed.

**What holds.** `run_command` is the only way to execute anything: NOOA blocks
`subprocess`, `socket` and the rest of the event-loop hazards at import *and*
strips them from the cell's namespace, and blocks `os.system` / `os.popen` by
name. So the allowlist, the denied flags, the path confinement, the process-group
kill and the output clipping in `tools/verify.py` are on the only path there is.

**What does not.** A cell has `open()`, so "read-only" is not a property this arm
can claim about any part of a run — which is why there are no read-only phases
here, and why the write guard reads the *tool log* rather than trusting a phase
boundary. A cell can also reach `self`, including hidden fields, so
`self.repo.repo_root` is a thing a determined model could rewrite. NVIDIA
documents these lists as guardrails and not a jail, and says so first:
containment is the OS-level sandbox.

For the benchmark that is fine, because the benchmark already runs each instance
in a container and grades from a fresh one. It is stated here because "only
Editing can write" is a sentence the other three arms can say about their
phase graph and this one cannot say about anything.

## SWE-bench

Wired into the shared bridge as `--framework nooa`:

```bash
cd ../CodeAgent/swebench_bridge
python run_agent.py --framework nooa \
    --instances astropy__astropy-12907 --run-id smoke --model gpt-5
python grade.py --predictions results/smoke/predictions.jsonl
python report.py results/smoke
```

or all four sides over one pinned instance set:

```bash
python compare.py --run-id four-way \
    --frameworks byllm langgraph openai nooa \
    --instances-file ../case_study/instances.txt
```

The bridge hands every framework the same workspace, container, objective text
and patch extraction; the only framework-specific thing is which shim gets
spawned. This side reuses `swe_entry.py` unchanged — it resolves the agent from
`$CODEAGENT_HOME` and imports `solve` / `active_model_name` / `DEFAULT_MODEL`
from `orchestrator`, which this project exports with byLLM's field set. Its
entry in `frameworks.py` is six lines, which is what the registry being the whole
of the fork is supposed to mean.

Note that `--framework nooa` needs `nooa` installed in the interpreter given by
`--python`, the same way `--framework langgraph` needs `langgraph`.

## Tests

```bash
python -m pytest tests -q
```

`tests/conftest.py` scripts `nooa.unifiedllm.FakeLLMClient` with
`execute_python` cells, so the whole agent runs — the real session, the real
REPL, the real tools, on a real temporary git repository — with nothing leaving
the process. `test_agent.py` covers the action space and each of the three
guards; `test_orchestrator.py` covers the report, the repair loop, the budget,
the downgrade, and the `RunResult` shape the shim reads.
