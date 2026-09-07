# Why Using Jac Reduces Developer Effort

A comparison of three **behavior-identical** implementations of the same five-phase SWE-bench coding agent, written respectively in **Jac** (`by llm`), **LangGraph** (`create_agent` + `StateGraph`), and the **raw OpenAI SDK**. We evaluate the developer effort each framework demands along two dimensions:

1. **Code conciseness** — less code to implement equivalent functionality.
2. **Directness of expression** — less abstraction and hand-written machinery to achieve the same functionality.

Every code excerpt below is verbatim from the three arms (`Jac/`, `langgraph/`, `openai_sdk/`).

## 1. Why the comparison is fair

The three arms are deliberately held **behavior-identical**. They share, byte-for-byte or word-for-word:

- The same 7-tool `Repo` toolbox. The tool method bodies are identical across arms; LangGraph only relocates the descriptions into Google-style docstrings, and Jac into `sem` statements.
- The same `SYSTEM` prompt, five phase goals, edge *capability* strings, and the two routing intents (`REPAIR_INTENT`, `RELOCATE_INTENT`).
- The same shared run-state and evidence predicates: `written()`, `ran_since_edit()`, `log_call`, `guard`, and the `TOOL_BUDGET` policy.
- The same graph and control policy: `Plan → Explore → Edit → Verify`, with `Verify` routing to `Edit` / `Explore` / `Finish` under identical deterministic guards.

Each README states this parity explicitly (*"share the repository tools, task instructions, phase transitions, evidence guards, and budget policy"*).

**Consequence:** every line that differs between the arms is *framework glue*, not tools, prompts, or policy. This isolates precisely the cost of expressing the LLM operations themselves.

The SWE-bench adapter (`orchestrator.*`: `RunResult` / `ToolCall`, the `solve` wrapper, token counting, per-call tracing) is excluded from all counts below, as it is shared plumbing unrelated to the agent logic.

## 2. Dimension 1 — Code conciseness

Developer-authored lines, excluding the SWE-bench adapter:

| Arm         | `main` | `nodes` | `phase_agent` | **Total** |
|-------------|-------:|--------:|--------------:|----------:|
| **Jac**     |    121 |     336 |             — |   **457** |
| **LangGraph** |  187 |     362 |             — |   **549** |
| **OpenAI SDK** |  117 |    549 |           313 |   **979** |

The identical `Repo` toolbox and its descriptions (~185–225 lines) appear in all three and cancel out. Subtracting them leaves the **framework-glue delta** — the code attributable purely to *how the framework expresses LLM operations*:

- **Jac ≈ 250 lines**
- **LangGraph ≈ 325 lines**
- **OpenAI SDK ≈ 765 lines** (≈ 3× Jac)

The OpenAI SDK arm requires an entire additional file, `phase_agent.py` (313 lines), that has **no counterpart in the Jac arm**. It exists solely to re-implement machinery that Jac obtains from the language.

## 3. Dimension 2 — Directness of expression

Four constructs carry essentially the entire gap. For each, the exact code from all three arms follows.

### A. The per-phase ReAct loop

In Jac, a phase's agentic tool-loop **is** the function signature. `by llm(...)` *is* the loop: build the request, parse tool calls, dispatch them, append results, repeat, and force a final answer at the budget.

**Jac** — `nodes.jac` (one clause per phase):

```jac
node Plan {
    has goal: str = "Restate the issue as one falsifiable claim (what call misbehaves, what it should do). Use grep/outline to name the 1-3 source files most likely involved. Do not edit.";
    def work(directive: str) -> str by llm(
        tools=[repo.grep, repo.outline, repo.read_file],
        conversation=history, max_react_iterations=6, on_iteration=guard
    );
}
```

**LangGraph** — `main.py` (the loop comes from `create_agent`, but the wrapper, budget middleware, and conversation writeback are hand-written):

```python
def work(phase: Phase) -> str:
    rounds = 0

    @wrap_model_call
    def budget(request, handler):
        nonlocal rounds
        # The shared budget is checked between rounds, after the first call.
        if rounds and (0 < phase.max_react_iterations <= rounds or len(tool_log) >= TOOL_BUDGET):
            request = request.override(tools=[], messages=[*request.messages, HumanMessage(FINAL_INSTRUCTION)])
        rounds += 1
        return handler(request)

    agent = create_agent(get_model(), tools=phase.toolbox(), middleware=[budget, tool_errors])
    result = agent.invoke(
        {"messages": [*history, HumanMessage(f"{phase.sem}\n\n{phase.goal}")]},
        {"max_concurrency": 1, "recursion_limit": 2 * phase.max_react_iterations + 10},
    )
    history[:] = result["messages"]
    return history[-1].text.strip()
```

**OpenAI SDK** — `phase_agent.py` (the entire loop is hand-written):

```python
def work(phase: Phase) -> str:
    """`here.work(here.goal)`: one phase, on the shared conversation."""
    tools = phase.toolbox()
    registry = {t.name: t for t in tools}
    specs = [t.spec() for t in tools] + [FINISH_TOOL]
    scaffold = {"role": "system", "content": SYSTEM_PERSONA + INSTRUCTION_TOOL}
    messages: list[dict[str, Any]] = [scaffold, *history, {"role": "user", "content": call_frame(phase)}]
    scaffolding: set[int] = {id(scaffold)}

    iteration = 0
    last_tool = ""
    last_result = ""
    while True:
        iteration += 1
        if iteration > 1 and guard(iteration, last_tool, last_result) == "abort_with_summary":
            return _force_final_answer(messages, scaffolding)
        if phase.max_react_iterations > 0 and iteration > phase.max_react_iterations:
            return _force_final_answer(messages, scaffolding)

        message = llm.complete(messages, tools=specs)
        messages.append(assistant_turn(message))
        calls = list(getattr(message, "tool_calls", None) or [])
        if not calls:
            # Plain text with no tool call ends the phase, as byLLM's str-return path does.
            _persist(messages, scaffolding)
            return (message.content or "").strip()

        finish = None
        for call in calls:
            if call.function.name == "finish_tool":
                finish = call
                break
            result = dispatch(registry, call.function.name, call.function.arguments)
            messages.append(tool_turn(call.id, call.function.name, result))
            last_tool = call.function.name
            last_result = result[:LAST_RESULT_CHARS]
        if finish is not None:
            output = _finish_output(finish.function.arguments)
            messages.append(tool_turn(finish.id, "finish_tool", output))
            _persist(messages, scaffolding)
            return output
```

…and this depends on hand-written helpers that Jac never exposes to the developer — `assistant_turn`, `tool_turn`, `parse_args`, `dispatch`, plus `_persist` and `_force_final_answer`:

```python
def assistant_turn(message: Any) -> dict[str, Any]:
    turn: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    calls = getattr(message, "tool_calls", None) or []
    if calls:
        turn["tool_calls"] = [
            {"id": c.id, "type": "function",
             "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in calls
        ]
    return turn

def tool_turn(call_id: str, name: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}

def dispatch(registry: dict[str, Tool], name: str, raw: str) -> str:
    tool = registry.get(name)
    if tool is None:
        return f"Error: no tool named '{name}'. Available: {', '.join(sorted(registry)) or '(none)'}."
    args, refusal = parse_args(raw)
    return refusal or tool.invoke(args)
```

### B. Exposing a tool to the model

**Jac** — pass the typed method and annotate its meaning with `sem`; the JSON schema is derived from the signature automatically:

```jac
def work(directive: str) -> str by llm(
    tools=[repo.grep, repo.outline, repo.read_file], ...
);

sem Repo.grep = "Search file contents with a Python regex. Returns 'path:line: text' per match. Use this to find where a symbol is defined or used.";
sem Repo.grep.pattern = "Python regular expression.";
sem Repo.grep.path = "File or directory to search, relative to the repository root.";
sem Repo.grep.file_glob = "Glob restricting file names, e.g. '*.py'.";
```

**LangGraph** — near-automatic via `StructuredTool.from_function`, but the descriptions must be encoded as Google-style docstrings on every method:

```python
def toolbox(self) -> list[BaseTool]:
    return [StructuredTool.from_function(getattr(repo, name), parse_docstring=True)
            for name in self.tools]

# every tool method must carry a docstring in this exact dialect:
def grep(self, pattern: str, path: str = ".", file_glob: str = "*.py") -> str:
    """Search file contents with a Python regex. Returns 'path:line: text' per match. Use this to find where a symbol is defined or used.

    Args:
        pattern: Python regular expression.
        path: File or directory to search, relative to the repository root.
        file_glob: Glob restricting file names, e.g. '*.py'.
    """
    ...
```

**OpenAI SDK** — fully manual: a `SEM` dict, a `_JSON_TYPES` map, and `make_tool` building the schema from `inspect.signature` + `get_type_hints`, plus a `Tool` dataclass with `spec()` and hand-rolled argument coercion in `invoke()`:

```python
SEM: dict[str, tuple[str, dict[str, str]]] = {
    "grep": (
        "Search file contents with a Python regex. Returns 'path:line: text' per match. Use this to find where a symbol is defined or used.",
        {"pattern": "Python regular expression.",
         "path": "File or directory to search, relative to the repository root.",
         "file_glob": "Glob restricting file names, e.g. '*.py'."}),
    # ... one entry per tool ...
}

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}

def make_tool(fn: Callable[..., str]) -> Tool:
    name = fn.__name__
    description, param_sem = SEM[name]
    props: dict[str, Any] = {}
    hints = get_type_hints(fn)
    for pname, param in inspect.signature(fn).parameters.items():
        props[pname] = {"type": _JSON_TYPES.get(hints.get(pname, str), "string"),
                        "description": param_sem.get(pname, "")}
    return Tool(name=name, fn=fn, description=f"{name}: {description}", parameters={
        "type": "object", "properties": props, "required": list(props),
        "additionalProperties": False})

@dataclass(frozen=True)
class Tool:
    name: str
    fn: Callable[..., str]
    description: str
    parameters: dict[str, Any]

    def spec(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}

    def invoke(self, args: dict[str, Any]) -> str:
        kwargs: dict[str, Any] = {}
        props = self.parameters["properties"]
        for key, value in args.items():
            if key not in props:
                continue
            want = props[key].get("type")
            try:
                if want == "integer" and not isinstance(value, bool):
                    value = int(value)
                elif want == "string" and not isinstance(value, str):
                    value = str(value)
            except (TypeError, ValueError):
                return f"Error: argument '{key}' of {self.name} must be {want}; got {value!r}."
            kwargs[key] = value
        try:
            out = self.fn(**kwargs)
        except Exception as e:
            return f"Error: {self.name} failed: {type(e).__name__}: {e}"
        return out if isinstance(out, str) else str(out)
```

### C. LLM-routed branching

In Jac, letting the model choose the next edge is a single expression embedded directly in graph traversal.

**Jac** — `main.jac` (inside `can verifying with Verify entry`):

```jac
visit [edge ->:PhaseCapability:kind != "relocate":->] by llm(
    select=1,
    intent="Pick the edge whose capability matches the state of the work. Take the 'finish' edge only if the repository's own tests covering the changed code ran after the last edit and passed, and the reproduction now passes. Otherwise take the 'repair' edge.",
    incl_info={"progress": "\n\n".join(self.ledger)}
) else {
    visit [->:PhaseCapability:->][?:Edit];
}
```

**LangGraph** — `main.py` (build the prompt, a `Literal` schema via `create_model`, `with_structured_output`, retry, and the fallback):

```python
def select_edge(candidates: list[PhaseCapability], intent: str, incl_info: dict,
                walker: dict, here: str) -> PhaseCapability | None:
    prompt = (f"{intent}\n\nAgent state: {walker}\nCurrent phase: {PHASES[here].goal}\n"
              f"Candidates: {[asdict(edge) for edge in candidates]}\nContext: {incl_info}\n"
              "Choose one of the candidate targets.")
    schema = create_model("Route", target=(Literal[tuple(edge.target for edge in candidates)], ...))
    router = get_model().with_structured_output(schema, method="function_calling").with_retry(
        retry_if_exception_type=(OutputParserException, ValidationError),
        stop_after_attempt=2, wait_exponential_jitter=False,
    )
    try:
        choice = router.invoke(prompt)
        return next((edge for edge in candidates if edge.target == choice.target), None)
    except Exception:
        return None
```

**OpenAI SDK** — `phase_agent.py` (`select_edge` + a hand-built JSON schema in `route_schema`, a schema hint, a `correction` retry message, and `parse_choice`):

```python
def select_edge(candidates, intent, incl_info, walker, here) -> PhaseCapability | None:
    if not candidates:
        return None
    handles = [e.target for e in candidates]
    lines = [f"{h}) here --({describe_edge(e)})--> {describe_node(e.target)}"
             for h, e in zip(handles, candidates)]
    parts = [f"Goal: {intent}", f"Walker:\n{describe_walker(walker)}",
             f"Current node:\n{describe_node(here)}",
             "Candidates (choose by handle):\n" + "\n".join(lines)]
    if incl_info:
        parts.append("\n".join(f"{k} = {v}" for k, v in incl_info.items()))
    parts.append(SCHEMA_HINT)
    response_format = route_schema(handles)
    messages = [{"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": "\n\n".join(parts)}]
    try:
        for attempt in range(2):
            message = llm.complete(messages, response_format=response_format)
            text = message.content or ""
            try:
                picked = parse_choice(text, handles)
            except ValueError as e:
                if attempt == 0:
                    messages.append({"role": "user", "content": correction(text, str(e), response_format)})
                    continue
                return None
            for item in picked:
                return candidates[handles.index(item)]
            return None
    except Exception:
        return None
    return None

def route_schema(handles: list[str]) -> dict[str, Any]:
    names = ", ".join(handles)
    return {"type": "json_schema", "json_schema": {"name": "list", "schema": {
        "type": "object", "title": "schema_object_wrapper",
        "properties": {"schema_object_wrapper": {
            "type": "array",
            "items": {"description": f"\nThe value *should* be one in this list: {handles!r} where the names are [{names}].",
                      "type": "string", "enum": list(handles)},
            "title": "List"}},
        "required": ["schema_object_wrapper"], "additionalProperties": False}, "strict": True}}
```

### D. Graph construction and control flow

**Jac** — native topology operators build the graph; on-entry abilities drive it with `visit`; the walker's `has` fields *are* the state the router sees:

```jac
can setup with Root entry {
    if not [-->] {
        pl = Plan(); ex = Explore(); ed = Edit(); ve = Verify(); fi = Finish();
        here ++> pl;
        pl +>: PhaseCapability(capability="locate the exact code to change and reproduce the failure") :+> ex;
        ex +>: PhaseCapability(capability="apply the fix to library source and re-run the reproduction") :+> ed;
        ed +>: PhaseCapability(capability="run the repository's own tests against the edited tree") :+> ve;
        ve +>: PhaseCapability(kind="repair", capability="repair: ...") :+> ed;
        ve +>: PhaseCapability(kind="relocate", capability="relocate: ...") :+> ex;
        ve +>: PhaseCapability(kind="finish", capability="finish: ...") :+> fi;
    }
    repo.base_dir = self.repo_root;
    history.clear();
    tool_log.clear();
    history.append({"role": "system", "content": SYSTEM.format(issue=self.issue)});
    visit [-->][?:Plan];
}

can planning with Plan entry {
    self.record("Plan", here.work(here.goal));
    visit [->:PhaseCapability:->][?:Explore];
}
```

**LangGraph** 

**OpenAI SDK** — degraded to a `while` loop dispatching on a string `here`, with state threaded manually through a dataclass:

```python
def walk(agent: CodeAgent) -> None:
    """`root spawn agent`."""
    repo.base_dir = agent.repo_root
    history.clear()
    tool_log.clear()
    history.append({"role": "system", "content": SYSTEM.format(issue=agent.issue)})
    here = "Plan"
    while True:
        if here == "Finish":
            repo.cleanup()
            agent.answer = agent.progress()
            return
        phase = PHASES[here]
        agent.record(here, work(phase))
        here = route_verify(agent) if here == "Verify" else _forward(here)
```

### Summary of the four constructs

| Construct     | Jac                                  | LangGraph                                    | OpenAI SDK                                        |
|---------------|--------------------------------------|----------------------------------------------|--------------------------------------------------|
| ReAct loop    | `by llm(...)` clause (1 / phase)     | `create_agent` + `work()` wrapper (~19 LOC)  | Hand-written loop + helpers (~90 LOC)            |
| Tool → schema | typed method + `sem`                 | `StructuredTool.from_function` + docstrings  | `SEM` + `make_tool` + `Tool` (~90 LOC)          |
| LLM routing   | `visit [edge ...] by llm(select=1)`  | `select_edge` + `create_model` (~16 LOC)     | `select_edge` + schema/hint/retry (~80 LOC)     |
| Graph + state | topology operators + walker `has`    | `StateGraph` + `AgentState` TypedDict        | `while` loop + manual state threading           |

## 4. Takeaway

Jac collapses four things the other frameworks make explicit — the **ReAct loop**, **tool-schema generation**, **structured LLM routing**, and **conversation state** — into language constructs (`by llm`, `sem`, `visit … by llm`). The OpenAI SDK arm needs an entire 313-line file with no Jac counterpart just to reconstitute them; even LangGraph, which supplies the loop and routing as library calls, still costs ~30% more glue and forces the developer into its schema and message abstractions.

Because the tools, prompts, and policy are held identical across all three arms, this difference is attributable entirely to *how directly each framework lets the developer express the LLM operations* — establishing both the conciseness (≈3× fewer glue lines than the raw SDK) and the directness (four hand-written subsystems replaced by four language constructs) that Jac's built-in LLM support provides.
