"""The per-phase ReAct loop.

byLLM gets this from one `by` clause:

    def run_phase(directive: str) -> str by agent_model(
        tools=..., max_react_iterations=25, on_iteration=progress_guard,
        max_tool_result_length=4000
    );

LangGraph has no single equivalent, so the loop is built here as a small
compiled StateGraph. Four behaviours have to be reproduced for the two sides to
stay comparable:

  * a hard 25-iteration cap
  * abort-with-summary when the model repeats the same call three times running
  * tool results clipped to 4000 characters
  * mutating tool calls never running concurrently

The cap and the guard both route to `summarize` rather than raising, because a
phase that ends with no text contributes nothing to the ledger and the next
phase then works blind. `recursion_limit` cannot do this -- it raises
GraphRecursionError.
"""

from __future__ import annotations

from typing import Annotated, Any, Callable, Sequence, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from telemetry import TokenUsage
from tools.common import clip

MAX_REACT_ITERATIONS: int = 25
MAX_TOOL_RESULT_CHARS: int = 4000
REPEAT_ABORT_THRESHOLD: int = 3

# Tools that must never run concurrently with anything else. Two concurrent
# writes, or a run_command racing a write, make a run unreproducible -- fatal
# for an A/B benchmark. This is the analogue of byLLM's `mark_serialize`.
SERIALIZED_TOOLS: frozenset[str] = frozenset(
    {"write_file", "replace_in_file", "run_command"}
)

ABORT_DIRECTIVE = (
    "Stop calling tools now. Reply with a short summary of what you did in this "
    "phase and what the next phase needs to know."
)

UNANSWERED_TOOL_CALL = "(not run: the phase reached its iteration limit)"


class ReactState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    recent_sigs: list[str]


def batch_signature(messages: Sequence[BaseMessage]) -> str:
    """One string standing for "what just happened".

    byLLM's IterationContext carries a single (last_tool, last_result) pair; a
    LangGraph turn can carry a batch, so the whole batch folds into one
    signature. Identical batches three times running means no progress.
    """
    parts = [
        f"{m.name}|{m.content}" for m in messages if isinstance(m, ToolMessage)
    ]
    return "\n".join(parts)


def progress_guard(state: ReactState) -> str:
    """Ends a phase that has stopped making progress.

    No legitimate workflow issues the identical call three times running. Kept a
    module-level function of the state so the tests can drive it directly.
    """
    sigs = state.get("recent_sigs") or []
    if len(sigs) >= REPEAT_ABORT_THRESHOLD:
        tail = sigs[len(sigs) - REPEAT_ABORT_THRESHOLD:]
        if all(s == tail[0] for s in tail):
            return "summarize"
    return "model"


def route_after_model(state: ReactState) -> str:
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return END
    # The model still wants tools but the budget is gone: let it summarise
    # rather than cutting the phase off mid-thought.
    if state.get("iterations", 0) >= MAX_REACT_ITERATIONS:
        return "summarize"
    return "tools"


class SerialToolNode:
    """Runs a batch of tool calls, serializing any batch that mutates.

    `langgraph.prebuilt.ToolNode` dispatches a batch through `executor.map`, so a
    turn that asks for `write_file` and `run_command` together runs them
    concurrently. Read-only batches keep that parallelism; a batch naming any
    tool in SERIALIZED_TOOLS is instead fed to the same ToolNode one call at a
    time, which serializes it while keeping ToolNode's error handling.
    """

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        self.node = ToolNode(list(tools))

    def __call__(self, state: ReactState) -> dict[str, Any]:
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", None) or [])
        if any(c["name"] in SERIALIZED_TOOLS for c in calls) and len(calls) > 1:
            produced: list[BaseMessage] = []
            for call in calls:
                stub = AIMessage(content="", tool_calls=[call])
                out = self.node.invoke({"messages": [stub]})
                produced.extend(out["messages"])
        else:
            produced = list(self.node.invoke({"messages": state["messages"]})["messages"])
        # byLLM's max_tool_result_length, applied at the same boundary.
        for msg in produced:
            if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
                msg.content = clip(msg.content, MAX_TOOL_RESULT_CHARS)
        sigs = list(state.get("recent_sigs") or [])
        sigs.append(batch_signature(produced))
        return {"messages": produced, "recent_sigs": sigs}


def build_phase_agent(
    tools: Sequence[BaseTool],
    model_ref: Callable[[], BaseChatModel],
    usage: TokenUsage,
) -> Any:
    """Compile the ReAct loop for one phase, over exactly that phase's tools.

    The model arrives as a zero-argument callable rather than an instance so
    that compiling the graph -- and so inspecting the topology -- never
    constructs a provider client, which would demand credentials.
    """
    bound_cache: list[Any] = []

    def bound() -> Any:
        if not bound_cache:
            bound_cache.append(model_ref().bind_tools(list(tools)))
        return bound_cache[0]

    def call_model(state: ReactState) -> dict[str, Any]:
        response = bound().invoke(state["messages"])
        usage.track(response)
        return {
            "messages": [response],
            "iterations": state.get("iterations", 0) + 1,
        }

    def force_summary(state: ReactState) -> dict[str, Any]:
        """byLLM's IterationAction.ABORT_WITH_SUMMARY."""
        messages = list(state["messages"])
        # A tool call left unanswered is an invalid conversation for the
        # provider, so close any that the iteration cap interrupted.
        last = messages[-1]
        pending: list[BaseMessage] = []
        if isinstance(last, AIMessage) and last.tool_calls:
            pending = [
                ToolMessage(
                    content=UNANSWERED_TOOL_CALL,
                    name=call["name"],
                    tool_call_id=call["id"],
                )
                for call in last.tool_calls
            ]
        directive = HumanMessage(content=ABORT_DIRECTIVE)
        # The unbound model: offering tools here just invites another call.
        response = model_ref().invoke(messages + pending + [directive])
        usage.track(response)
        return {"messages": pending + [directive, response]}

    graph = StateGraph(ReactState)
    graph.add_node("model", call_model)
    graph.add_node("tools", SerialToolNode(tools))
    graph.add_node("summarize", force_summary)
    graph.set_entry_point("model")
    graph.add_conditional_edges(
        "model",
        route_after_model,
        {"tools": "tools", "summarize": "summarize", END: END},
    )
    graph.add_conditional_edges(
        "tools",
        progress_guard,
        {"model": "model", "summarize": "summarize"},
    )
    graph.add_edge("summarize", END)
    return graph.compile()


def last_text(messages: Sequence[BaseMessage]) -> str:
    """The final thing the model actually said, for the phase ledger."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                # Some providers return content blocks rather than a string.
                content = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                )
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def run_phase(
    agent: Any,
    messages: Sequence[BaseMessage],
) -> tuple[str, list[BaseMessage]]:
    """Drive one phase to completion; return its summary and its conversation."""
    out = agent.invoke(
        {"messages": list(messages), "iterations": 0, "recent_sigs": []},
        # A backstop only: the loop ends itself at MAX_REACT_ITERATIONS.
        {"recursion_limit": 2 * MAX_REACT_ITERATIONS + 5},
    )
    return last_text(out["messages"]), list(out["messages"])


__all__: list[str] = [
    "MAX_REACT_ITERATIONS",
    "MAX_TOOL_RESULT_CHARS",
    "REPEAT_ABORT_THRESHOLD",
    "SERIALIZED_TOOLS",
    "ReactState",
    "SerialToolNode",
    "batch_signature",
    "build_phase_agent",
    "last_text",
    "progress_guard",
    "route_after_model",
    "run_phase",
]
