"""What one `by llm(...)` clause does on the Jac side, as a compiled StateGraph.

    def work(directive: str) -> str by llm(
        tools=[...], conversation=history, max_react_iterations=N, on_iteration=guard
    );

byLLM's ReAct loop for that clause (jaclang/byllm, non-streaming):

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

Here that is a three-node graph, `model -> tools -> (model | summarize)`, and
the write-back filters on a `scaffolding` marker because `add_messages` copies
messages and identity does not survive it. `select_edge` is the walker's other
`by llm` form, `visit [...] by llm(select=1, intent=..., incl_info=...)`: one
fresh structured-output call over the candidate handles.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from nodes import (
    CAPABILITY_SEM,
    DIRECTIVE_SEM,
    FINISH_TOOL,
    PHASES,
    PHASE_CAPABILITY_SEM,
    Phase,
    PhaseCapability,
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
SCAFFOLDING = "byllm_scaffolding"


class ReactState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    last_tool: str
    last_result: str
    finished: bool
    output: str


def call_frame(phase: Phase) -> str:
    """The user message byLLM renders for `node.work(directive=goal)`."""
    header = f"{phase.name}.work(directive: str) -> str --- {phase.sem}"
    return f"{header}\n      directive: str ---- {DIRECTIVE_SEM}\ndirective = {phase.goal!r}"


def _scaffold(message: BaseMessage) -> BaseMessage:
    message.additional_kwargs[SCAFFOLDING] = True
    return message


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content or "").strip()


def _persist(messages: list[BaseMessage]) -> None:
    """MTRuntime.write_back_conversation: everything but byLLM's own scaffolding.

    The finish_tool exchange is dropped as byLLM drops it; a mixed batch keeps
    its other calls, so no persisted call is ever left without its result.
    """
    out: list[BaseMessage] = []
    for m in messages:
        if m.additional_kwargs.get(SCAFFOLDING):
            continue
        if isinstance(m, ToolMessage) and m.name == "finish_tool":
            continue
        if isinstance(m, AIMessage) and m.tool_calls:
            kept = [c for c in m.tool_calls if c["name"] != "finish_tool"]
            if not kept:
                continue
            if len(kept) != len(m.tool_calls):
                m = AIMessage(content=m.content, tool_calls=kept, id=m.id)
        out.append(m)
    history[:] = out


def build_phase_agent(phase: Phase) -> Any:
    """Compile the loop for one phase over exactly that phase's tools."""
    tools = phase.toolbox()
    runner = ToolNode(tools)
    bound = llm.chat().bind_tools(tools + [FINISH_TOOL])
    finish_only = llm.chat().bind_tools([FINISH_TOOL])

    def call_model(state: ReactState) -> dict[str, Any]:
        response = bound.invoke(state["messages"])
        llm.track(response)
        update: dict[str, Any] = {"messages": [response], "iterations": state["iterations"] + 1}
        if not response.tool_calls:
            # Plain text with no tool call ends the phase, as byLLM's str-return path does.
            update.update(finished=True, output=_text(response))
        return update

    def run_tools(state: ReactState) -> dict[str, Any]:
        """Sequential dispatch, split at the first finish_tool -- byLLM's batch rule."""
        last = state["messages"][-1]
        produced: list[BaseMessage] = []
        update: dict[str, Any] = {}
        for call in last.tool_calls:
            if call["name"] == "finish_tool":
                output = call["args"].get("final_output", "")
                output = output if isinstance(output, str) else str(output)
                produced.append(ToolMessage(content=output, name="finish_tool", tool_call_id=call["id"]))
                update.update(finished=True, output=output)
                break
            stub = AIMessage(content="", tool_calls=[call])
            out = runner.invoke({"messages": [stub]})["messages"]
            produced.extend(out)
            for m in out:
                if isinstance(m, ToolMessage):
                    update["last_tool"] = str(m.name)
                    update["last_result"] = str(m.content)[:LAST_RESULT_CHARS]
        update["messages"] = produced
        return update

    def force_final_answer(state: ReactState) -> dict[str, Any]:
        """IterationAction.ABORT_WITH_SUMMARY and the iteration cap: one more call, finish_tool only."""
        nudge = _scaffold(HumanMessage(content=FINAL_INSTRUCTION))
        response = finish_only.invoke(state["messages"] + [nudge])
        llm.track(response)
        produced: list[BaseMessage] = [nudge, response]
        output = _text(response)
        for call in response.tool_calls:
            if call["name"] == "finish_tool":
                output = call["args"].get("final_output", "")
                output = output if isinstance(output, str) else str(output)
                produced.append(ToolMessage(content=output, name="finish_tool", tool_call_id=call["id"]))
                break
        return {"messages": produced, "finished": True, "output": output}

    def after_model(state: ReactState) -> str:
        return END if state["finished"] else "tools"

    def after_tools(state: ReactState) -> str:
        if state["finished"]:
            return END
        # The next round's pre-flight, in byLLM's order: on_iteration, then the cap.
        if guard(state["iterations"] + 1, state["last_tool"], state["last_result"]) == "abort_with_summary":
            return "summarize"
        if phase.max_react_iterations > 0 and state["iterations"] + 1 > phase.max_react_iterations:
            return "summarize"
        return "model"

    graph = StateGraph(ReactState)
    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)
    graph.add_node("summarize", force_final_answer)
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", after_model, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", after_tools, {"model": "model", "summarize": "summarize", END: END})
    graph.add_edge("summarize", END)
    return graph.compile()


_agents: dict[str, Any] = {}


def work(phase: Phase) -> str:
    """`here.work(here.goal)`: one phase, on the shared conversation."""
    agent = _agents.get(phase.name)
    if agent is None:
        agent = _agents[phase.name] = build_phase_agent(phase)
    scaffold = _scaffold(SystemMessage(content=SYSTEM_PERSONA + INSTRUCTION_TOOL))
    opening: list[BaseMessage] = [scaffold, *history, HumanMessage(content=call_frame(phase))]
    out = agent.invoke(
        {"messages": opening, "iterations": 0, "last_tool": "", "last_result": "",
         "finished": False, "output": ""},
        # A backstop only: the loop ends itself at the phase's own cap.
        {"recursion_limit": 2 * max(phase.max_react_iterations, 1) + 10},
    )
    _persist(list(out["messages"]))
    return str(out["output"] or "")


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
    return (f"{PHASE_CAPABILITY_SEM}(capability ({CAPABILITY_SEM})={_safe_repr(edge.capability)}, "
            f"kind={_safe_repr(edge.kind)})")


def select_edge(
    candidates: list[PhaseCapability],
    intent: str,
    incl_info: dict[str, Any],
    walker: dict[str, Any],
    here: str,
) -> PhaseCapability | None:
    """One routing call; the chosen edge, or None when the model picked nothing usable."""
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
    # byLLM's inject_schema_hint: the enum constraint restated in the last user
    # message, because ollama-style servers treat response_format as a grammar
    # at best and answer in prose without it.
    parts.append(schema_hint(handles))
    schema = {"type": "object",
              "properties": {"result": {"type": "array", "items": {"type": "string", "enum": handles}}},
              "required": ["result"], "additionalProperties": False}
    try:
        router = llm.chat().bind(response_format={
            "type": "json_schema", "json_schema": {"name": "RouteChoice", "strict": True, "schema": schema}})
        response = router.invoke([SystemMessage(content=ROUTER_SYSTEM),
                                  HumanMessage(content="\n\n".join(parts))])
        llm.track(response)
        text = _text(response)
    except Exception:  # noqa: BLE001 - a routing failure is the `else` branch, not a crash
        return None
    for item in parse_choice(text, handles):
        return candidates[handles.index(item)]
    return None


def schema_hint(handles: list[str]) -> str:
    return "Output must be a JSON object matching the schema.\n- result must be one of: " + ", ".join(handles)


def parse_choice(text: str, handles: list[str]) -> list[str]:
    """The handles the answer names, JSON first, then the handles as bare words."""
    try:
        data = json.loads(text)
        chosen = data.get("result") if isinstance(data, dict) else data
        if isinstance(chosen, str):
            chosen = [chosen]
        picked = [c for c in (chosen or []) if isinstance(c, str) and c in handles]
        if picked:
            return picked
    except ValueError:
        pass
    found = [(m.start(), h) for h in handles for m in re.finditer(rf"\b{re.escape(h)}\b", text)]
    return [h for _, h in sorted(found)]


__all__ = [
    "FINAL_INSTRUCTION", "INSTRUCTION_TOOL", "ROUTER_SYSTEM", "SYSTEM_PERSONA", "ReactState",
    "build_phase_agent", "call_frame", "select_edge", "work",
]
