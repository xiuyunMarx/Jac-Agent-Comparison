"""The per-phase ReAct loop.

byLLM gets this from one `by` clause:

    def run_phase(directive: str) -> str by agent_model(
        tools=..., max_react_iterations=25, on_iteration=progress_guard,
        max_tool_result_length=MAX_TOOL_RESULT_CHARS
    );

LangGraph has no single equivalent, so the loop is built here as a small
compiled StateGraph. Four behaviours have to be reproduced for the two sides to
stay comparable:

  * a hard 25-iteration cap
  * abort-with-summary when the model repeats the same call three times running
  * abort-with-summary when one tool is reached for six times with nothing in
    between -- the identical-call test cannot see this one, because read_file
    walking a file issues a different call every time
  * tool results clipped to MAX_TOOL_RESULT_CHARS, and compared over only the
    first SIG_RESULT_CHARS of that when deciding "identical"
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
from tools.common import MAX_TOOL_CHARS, clip

MAX_REACT_ITERATIONS: int = 25
# Tied to the tools' own budget rather than set independently, and deliberately
# not set below it. Every tool already truncates its result, and does so in the
# terms the model can act on -- read_file stops on a line boundary and names the
# start_line that resumes it. A smaller cap here would cut that result a second
# time, mid-line and with no continuation hint, and the model has no way to tell
# a silently shortened result from a complete one: the only move a mystery
# truncation leaves is to repeat the identical call.
#
# This read 4000 until the three implementations were compared directly. It was
# the only place they disagreed on a tool's context budget, and it disagreed by
# 5x -- enough that a token gap between the frameworks measured this constant
# rather than the frameworks. byLLM and openai_sdk both tie it to MAX_TOOL_CHARS;
# this now does too.
MAX_TOOL_RESULT_CHARS: int = MAX_TOOL_CHARS
REPEAT_ABORT_THRESHOLD: int = 3
# One tool called over and over with nothing else between it is the other way a
# phase stalls, and the identical-call test above cannot see it: read_file
# walking a file one line at a time issues a different call every time, so every
# signature differs. Six is above any honest run of paging that grep could not
# have replaced, and the abort summarises rather than raising, so a false
# positive costs one phase ending early rather than the run.
SAME_TOOL_ABORT_THRESHOLD: int = 6
# How much of a tool result the repeat signature compares. byLLM hands its
# on_iteration hook a 500-character last_result, so comparing the full
# MAX_TOOL_RESULT_CHARS here would make this brake far less eager than byLLM's
# for the same conversation.
SIG_RESULT_CHARS: int = 500

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
    recent_tools: list[str]


def batch_signature(messages: Sequence[BaseMessage]) -> str:
    """One string standing for "what just happened".

    byLLM's IterationContext carries a single (last_tool, last_result) pair; a
    LangGraph turn can carry a batch, so the whole batch folds into one
    signature. Identical batches three times running means no progress.
    """
    parts = [
        f"{m.name}|{str(m.content)[:SIG_RESULT_CHARS]}"
        for m in messages
        if isinstance(m, ToolMessage)
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
    tools_seen = state.get("recent_tools") or []
    if len(tools_seen) >= SAME_TOOL_ABORT_THRESHOLD:
        tail = tools_seen[len(tools_seen) - SAME_TOOL_ABORT_THRESHOLD:]
        if all(t == tail[0] for t in tail):
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
        # One entry per executed tool, not per batch: byLLM's IterationContext
        # reports a single tool per iteration, and a batch of six identical
        # calls is the same stall as six batches of one.
        tools_seen = list(state.get("recent_tools") or [])
        tools_seen.extend(
            str(m.name) for m in produced if isinstance(m, ToolMessage)
        )
        return {
            "messages": produced,
            "recent_sigs": sigs,
            "recent_tools": tools_seen,
        }


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
        {
            "messages": list(messages),
            "iterations": 0,
            "recent_sigs": [],
            "recent_tools": [],
        },
        # A backstop only: the loop ends itself at MAX_REACT_ITERATIONS.
        {"recursion_limit": 2 * MAX_REACT_ITERATIONS + 5},
    )
    return last_text(out["messages"]), list(out["messages"])


__all__: list[str] = [
    "MAX_REACT_ITERATIONS",
    "MAX_TOOL_RESULT_CHARS",
    "SAME_TOOL_ABORT_THRESHOLD",
    "SIG_RESULT_CHARS",
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
