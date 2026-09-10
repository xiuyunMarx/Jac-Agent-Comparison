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
