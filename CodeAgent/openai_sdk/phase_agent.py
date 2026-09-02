"""What one `by llm(...)` clause does on the Jac side, written out.

    def work(directive: str) -> str by llm(
        tools=[...], conversation=history, max_react_iterations=N, on_iteration=guard
    );

byLLM's ReAct loop, as it runs for that clause (jaclang/byllm, non-streaming):

  * the request is [byLLM's own system message] + the caller's conversation +
    one user message rendering the call frame (`Plan.work(directive=...)`
    with the `sem` strings);
  * the phase's tools plus `finish_tool(final_output)`; the phase ends when the
    model calls finish_tool, or when it answers in plain text;
  * tool calls run one at a time, in order (byLLM's default is sequential);
  * `on_iteration` runs before every round after the first and can abort with
    a summary; so does exceeding `max_react_iterations`. Both ask the model
    once more, with only finish_tool, for "only your final answer";
  * everything but byLLM's own scaffolding (its system message, the
    final-answer nudge, the finish_tool call) is written back into the
    caller's conversation, so the next phase continues the same one.

`select_edge` is the other `by llm` form the walker uses:
`visit [edge ...] by llm(select=1, intent=..., incl_info=...)` -- one fresh
call, candidates rendered as byLLM's visit router renders them, a JSON-schema
answer naming one handle.
"""

from __future__ import annotations

import json
from typing import Any

from nodes import (
    DIRECTIVE_SEM,
    FINISH_TOOL,
    PHASES,
    Phase,
    PhaseCapability,
    Tool,
    guard,
    history,
    llm,
)

# jaclang/byllm/impl/mtir.impl.jac: SYSTEM_PERSONA and INSTRUCTION_TOOL.
SYSTEM_PERSONA = (
    "This is a task you must complete by returning only the output. The task will be "
    "expressed in the form of function call with arguments. Do not include explanations, "
    "code, or extra text—only the result."
)
INSTRUCTION_TOOL = (
    "Use the tools provided to reach the goal. Call one tool at a time with proper args—no "
    "explanations, no narration. Think step by step, invoking tools as needed. When done, "
    "always call finish_tool(output) to return the final output. Only use tools."
)
# BaseLLM._force_final_answer
FINAL_INSTRUCTION = "Based on the tool calls and their results above, provide only your final answer."
# visit_routing.jac
ROUTER_SYSTEM = (
    "You are routing a graph walker. Choose which candidate node(s) the walker should "
    "visit next, by handle. Return only valid handles. Choose exactly one."
)
LAST_RESULT_CHARS = 500


def call_frame(phase: Phase) -> str:
    """The user message byLLM renders for `node.work(directive=goal)`."""
    header = f"{phase.name}.work(directive: str) -> str --- {phase.sem}"
    frame = f"{header}\n      directive: str ---- {DIRECTIVE_SEM}\ndirective = {phase.goal!r}"
    # byLLM's identity zone: `self`, since the node carries a typed member.
    return f"{frame}\n\nself = {phase.name}(goal={phase.goal!r})\n    - goal: str"


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


def parse_args(raw: str) -> tuple[dict[str, Any], str]:
    if not raw or not raw.strip():
        return {}, ""
    try:
        parsed = json.loads(raw)
    except ValueError as e:
        return {}, f"Error: the arguments were not valid JSON ({e}). Re-issue the call with a JSON object."
    if not isinstance(parsed, dict):
        return {}, f"Error: the arguments must be a JSON object, not {type(parsed).__name__}."
    return parsed, ""


def dispatch(registry: dict[str, Tool], name: str, raw: str) -> str:
    tool = registry.get(name)
    if tool is None:
        return f"Error: no tool named '{name}'. Available: {', '.join(sorted(registry)) or '(none)'}."
    args, refusal = parse_args(raw)
    return refusal or tool.invoke(args)


def _finish_output(raw: str) -> str:
    args, _ = parse_args(raw)
    out = args.get("final_output", "")
    return out if isinstance(out, str) else json.dumps(out)


def _persist(messages: list[dict[str, Any]], scaffolding: set[int]) -> None:
    """MTRuntime.write_back_conversation: everything but byLLM's own scaffolding.

    The finish_tool exchange is dropped as byLLM drops it; a mixed batch keeps
    its other calls, so no persisted call is ever left without its result.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if id(m) in scaffolding:
            continue
        if m.get("role") == "tool" and m.get("name") == "finish_tool":
            continue
        calls = m.get("tool_calls") or []
        if calls:
            kept = [c for c in calls if c["function"]["name"] != "finish_tool"]
            if not kept:
                continue
            if len(kept) != len(calls):
                m = {**m, "tool_calls": kept}
        out.append(m)
    history[:] = out


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


def _force_final_answer(messages: list[dict[str, Any]], scaffolding: set[int]) -> str:
    """IterationAction.ABORT_WITH_SUMMARY and the iteration cap: one more call, finish_tool only."""
    nudge = {"role": "user", "content": FINAL_INSTRUCTION}
    scaffolding.add(id(nudge))
    messages.append(nudge)
    message = llm.complete(messages, tools=[FINISH_TOOL])
    turn = assistant_turn(message)
    messages.append(turn)
    output = ""
    for call in getattr(message, "tool_calls", None) or []:
        if call.function.name == "finish_tool":
            output = _finish_output(call.function.arguments)
            messages.append(tool_turn(call.id, "finish_tool", output))
            break
    else:
        output = (message.content or "").strip()
    _persist(messages, scaffolding)
    return output


# ---------------------------------------------------------------------------
# `visit [edge ...] by llm(select=1, intent=..., incl_info=...)`
# ---------------------------------------------------------------------------
def _safe_repr(value: Any, limit: int = 500) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    keep = (limit - 5) // 2
    return text[:keep] + " ... " + text[-keep:]


def describe_walker(fields: dict[str, Any]) -> str:
    return "CodeAgent(" + ", ".join(f"{k}={_safe_repr(v)}" for k, v in fields.items()) + ")"


def describe_node(phase_name: str) -> str:
    phase = PHASES[phase_name]
    return f"{phase.name}(goal={_safe_repr(phase.goal)})"


def describe_edge(edge: PhaseCapability) -> str:
    # visit_routing renders the archetype's repr: class name and fields, no sems.
    return f"PhaseCapability(capability={_safe_repr(edge.capability)}, kind={_safe_repr(edge.kind)})"


def select_edge(
    candidates: list[PhaseCapability],
    intent: str,
    incl_info: dict[str, Any],
    walker: dict[str, Any],
    here: str,
) -> PhaseCapability | None:
    """One routing call (two when the first answer is not JSON, as byLLM
    retries once); the chosen edge, or None when nothing usable came back."""
    if not candidates:
        return None
    handles = [e.target for e in candidates]      # node class names are unique here
    lines = [f"{h}) here --({describe_edge(e)})--> {describe_node(e.target)}"
             for h, e in zip(handles, candidates)]
    parts = [f"Goal: {intent}", f"Walker:\n{describe_walker(walker)}",
             f"Current node:\n{describe_node(here)}",
             "Candidates (choose by handle):\n" + "\n".join(lines)]
    if incl_info:
        parts.append("\n".join(f"{k} = {v}" for k, v in incl_info.items()))
    parts.append(SCHEMA_HINT)
    response_format = route_schema(handles)
    messages: list[dict[str, Any]] = [{"role": "system", "content": ROUTER_SYSTEM},
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
    except Exception:  # noqa: BLE001 - a routing failure is the `else` branch, not a crash
        return None
    return None


def route_schema(handles: list[str]) -> dict[str, Any]:
    """byLLM's response_format for `select=1` over these handles, field for field."""
    names = ", ".join(handles)
    return {"type": "json_schema", "json_schema": {"name": "list", "schema": {
        "type": "object", "title": "schema_object_wrapper",
        "properties": {"schema_object_wrapper": {
            "type": "array",
            "items": {"description": f"\nThe value *should* be one in this list: {handles!r} where the names are [{names}].",
                      "type": "string", "enum": list(handles)},
            "title": "List"}},
        "required": ["schema_object_wrapper"], "additionalProperties": False}, "strict": True}}


# byLLM's inject_schema_hint, as it renders for that schema: the field list,
# not the enum. GLM over ollama's /v1 answers the first call in prose, and it
# is the correction below that brings the JSON; the Jac arm pays both calls.
SCHEMA_HINT = "Schema requirements:\n- schema_object_wrapper (array)"


def correction(text: str, error: str, response_format: dict[str, Any]) -> str:
    """byLLM's retry message after an unparseable structured answer."""
    return ("Your previous response could not be used. The output parser reported: "
            f"Failed to convert LLM output to 'list': {error}.\n"
            f"Your previous response was:\n{text}\n"
            "You MUST reply with ONLY a single JSON object that validates against this JSON schema for `list`:\n"
            f"{json.dumps(response_format)}\n"
            "Output the raw JSON object only, with no explanation, no reasoning text, no markdown code fences, "
            "nothing before or after the JSON.\n\n" + SCHEMA_HINT)


def parse_choice(text: str, handles: list[str]) -> list[str]:
    """The handles the JSON answer names; raises ValueError as byLLM's parser does."""
    data = json.loads(text)
    chosen = data.get("schema_object_wrapper") if isinstance(data, dict) else data
    if isinstance(chosen, str):
        chosen = [chosen]
    if not isinstance(chosen, list):
        raise ValueError("not a list")
    return [c for c in chosen if isinstance(c, str) and c in handles]


__all__ = [
    "FINAL_INSTRUCTION", "INSTRUCTION_TOOL", "ROUTER_SYSTEM", "SYSTEM_PERSONA",
    "assistant_turn", "call_frame", "dispatch", "select_edge", "tool_turn", "work",
]
