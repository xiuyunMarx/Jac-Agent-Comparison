# Core vs adapter classification criterion (apply identically to every arm)

Goal: measure how much code a developer must write to express the AGENT ITSELF,
independent of the benchmark harness around it.  Every top-level symbol in every
file gets exactly one tag.

## CORE (count it) — "execution logic and state"
- State definitions: dataclasses, TypedDict, pydantic models, Jac `obj`/`node`/`edge`/`enum`,
  graph state classes, phase/step enums, result/answer types.
- Execution logic: graph/flow wiring (add_node/add_edge/conditional edges, walker
  `can` abilities, `visit`/`spawn`), node/phase functions, routing/decision functions,
  ReAct/tool-call loops, retry/budget/stop conditions, orchestrator classes,
  the run/invoke function that drives the agent over ONE input.
- Tools the model can call: the tool function bodies and their registration/wrapping
  (StructuredTool, @tool, @function_tool, Jac `def ... ` used as a tool, tool schemas).
- Prompts / semantics: Jac `sem` statements, `by llm()` declarations, prompt template
  strings, CrewAI agents.yaml / tasks.yaml, system-prompt constants, docstrings that
  the framework turns into prompts.
- LLM binding: construction of the Model/LLM client object the agent calls
  (e.g. `Model(...)`, `ChatOpenAI(...)`, `OpenAI(...)`, `get_chat_model`), and any
  hand-written call-the-model / parse-the-response helper that the agent needs
  because the framework does not do it (this is real developer effort).

## ADAPTER (exclude it)
- Imports, `from __future__`, module docstrings, `__all__`.
- Config plumbing: env-var reads, constants like MAX_ROUNDS = int(os.environ...),
  path constants, API base defaults.  (Exception: the Model/LLM object itself is core.)
- CLI / entry points: argparse, `if __name__ == "__main__"`, Jac `with entry {}`
  blocks whose job is to parse args and run over a dataset, batch loops over a
  dataset, reading input JSON files, writing result files, printing summaries.
- Benchmark/harness glue: mock mailbox / mock Slack / mock Trello / mock outputs,
  SWE-bench workspace bridges, "record for eval" hooks, result serialisation.
- Observability: tracing, telemetry, token/cost capture, timing, logging setup,
  trace_event/record_event helpers, debug dumps.
- Determinism aids added for benchmarking only: on-disk caches, simulated delays,
  seeded fakes.
- Tests, `__init__.py` re-exports.

## Rules
- Granularity is the top-level symbol (function/class/obj/walker/sem/etc.) as
  listed by `python3 loc_symbols.py <file>`.  Do not split a symbol; tag by its
  DOMINANT purpose.  If a function is 80% agent logic with a trace call inside,
  it is core.  If a function exists only to serve the harness, it is adapter even
  if it touches agent state.
- A class whose methods are mostly core is core (tag the whole class).  A class
  that is mostly telemetry/IO is adapter.
- Mirror decisions across arms: if "load_mailbox" is adapter in the Jac arm, the
  equivalent in the openai_sdk arm is adapter too.  Where a tool body embeds a
  cache or delay, keep the tool core (the tool is required).
- When unsure, write the reason and pick the tag; the reviewer will re-check
  anything with a `?` in the reason.

## PROTECTION code (third pass; `<app>.guard.json`, subtracted by `loc_count.py --core --no-guard`)

Goal: the agent's core logic with defensive code removed.  Protection = lines whose
only job is to detect, reject, repair or recover from bad model output, bad data,
or a misbehaving tool call.  Ranges are sub-symbol `[start, end, reason]`; only lines
already tagged core matter (adapter lines are gone anyway).

PROTECTION (subtract):
- Model-output validation: schema/isinstance/enum checks that raise or coerce,
  `_enum`-style coercers, empty-field validators, pydantic validators that reject
  or clamp, "no task list in the reply" style raises, fence stripping / lenient
  JSON extraction added because the model ignores response_format.
- Repair and retry: repair LLM methods and their prompts/sems, RETRY/correction
  prompts, the correction turn inside structured_call, try/except around a model
  call that re-asks, retry budgets (`max_output_retries`, `OUTPUT_ATTEMPTS`,
  `max_retries=` in a strategy config, `max_output_retries=` inside `by llm(...)`).
- Fallbacks: PARSE_FAILURE_REPLY-style texts, fallback_events bookkeeping, the
  branch that emits a canned answer when the tool loop or parse fails, skip/continue
  branches for malformed items ("missing a name or description. Skipping").
- Tool-safety guards: path confinement (safe_path, refuse_path), write-protected
  dirs, command allow/deny lists and flag screening, shell-metachar refusal,
  timeouts/kill/grace handling, output-size caps and truncation (clip/clip_ends,
  MAX_*_CHARS/BYTES), duplicate-call / no-progress detection, postcondition
  invariants (must_have_written / must_have_verified) and their feedback prompts.

NOT protection (keep):
- The main loop's iteration cap / round budget (MAX_ROUNDS, max_react_iterations,
  MAX_STEPS) and the normal stop condition: without them the loop is unbounded.
- Parsing that is the only way to read the reply at all (json.loads + field
  pickup); only the reject/raise/coerce lines around it are protection.
- Prompt text that merely asks for clean output ("never return an empty task").
- Rate limiting / caching of live HTTP fetches (bench determinism, already handled).
- Tool bodies themselves (read_file, grep, run_command execution) minus the guard lines.

Rules: mirror across arms (a guard inlined in a Jac ability and factored into a
Python helper get the same lines removed); brace/closing lines that only exist for
the removed block go with it; when in doubt keep the line and add `?` to the reason.
