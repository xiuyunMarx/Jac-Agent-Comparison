"""ReAct Agent to process messages with tool calls."""

from typing import Literal

import structlog
from django.conf import settings
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import (
    END,
    StateGraph,
)
from langgraph.graph.state import CompiledStateGraph

from app.schemas import (
    AgentOutput,
    AgentState,
)
from app.services.agent.llm import get_chat_model
from app.services.agent.prompts import SYSTEM_PROMPT_TEMPLATE
from app.services.agent.trace import record_event
from app.services.vector_database.tools import (
    SQLTools,
    VectorDatabaseTools,
)

logger = structlog.get_logger(__name__)

tools = [
    VectorDatabaseTools.tool(),
    SQLTools.tool(),
]

_model = None


def _get_model():
    """Build the tool-bound model on first use.

    Lazy so importing this module (Django loads it for every manage.py
    command) doesn't require LLM credentials.
    """
    global _model
    if _model is None:
        _model = get_chat_model(settings.POWERFUL_LLM, temperature=0.0).bind_tools(tools)
    return _model


tools_by_name = {tool.name: tool for tool in tools}


# Define our tool node
async def tool_node(state: AgentState, config: RunnableConfig = None) -> AgentState:
    """Process tool calls from the last message.

    Args:
        state: The current agent state containing messages and tool calls.
        config: The runnable config, propagated so callbacks see the tool calls.

    Returns:
        Dict with updated messages containing tool responses.
    """
    outputs = []
    for tool_call in state.messages[-1].tool_calls:
        tool_result = await tools_by_name[tool_call["name"]].ainvoke(tool_call["args"], config)
        outputs.append(
            ToolMessage(
                content=tool_result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


async def call_model(
    state: AgentState,
    config: RunnableConfig,
) -> AgentState:
    """Call the model with the system prompt and tool calls.

    Args:
        state: The current agent state containing messages and tool calls.
        config: The configuration for the runnable.

    Returns:
        Dict with updated messages containing the model's response.
    """
    output_parser = PydanticOutputParser(pydantic_object=AgentOutput)

    system_prompt = SystemMessage(
        SYSTEM_PROMPT_TEMPLATE.format(
            channel=await state.channel.pretty_str(),
            user=state.user,
            format_instructions=output_parser.get_format_instructions(),
        )
    )

    model = _get_model()
    response = await model.ainvoke([system_prompt] + conversation_window(state.messages), config)
    if not response.tool_calls:
        response = normalize_final_answer(response)
    return {"messages": [response]}


def conversation_window(messages: list) -> list:
    """The messages the model sees: the last three prior exchanges plus every
    message of the current turn.

    This used to be ``trim_messages(max_tokens=2000, start_on="human")``.
    With ``allow_partial=False`` that returns an empty list as soon as one
    transcript-sized tool result pushes the turn past the budget (the leading
    human message is dropped first, then ``start_on`` discards the rest), so
    the model received a system-only request and answered nothing at all. The
    other arms never trim inside a turn; prior turns are bounded the way the
    SDK arm bounds them (three exchanges), so the layout on the wire matches.
    """
    turn_start = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            turn_start = i
            break
    prior = [
        m
        for m in messages[:turn_start]
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)
    ]
    return prior[-6:] + [m for m in messages[turn_start:] if not isinstance(m, SystemMessage)]


def normalize_final_answer(response: AIMessage) -> AIMessage:
    """Store the final answer as canonical AgentOutput JSON.

    GLM 5.2 wraps the object in ```json fences or a sentence; the
    PydanticOutputParser reads through both. The non-tool path already did
    this (main_graph.non_tool_calls_reply); the tool path handed the raw text
    to the caller, and the benchmark's strict validation then failed on every
    fenced reply.
    """
    output_parser = PydanticOutputParser(pydantic_object=AgentOutput)
    raw = response.content if isinstance(response.content, str) else str(response.content)
    try:
        response.content = output_parser.parse(raw).model_dump_json()
    except OutputParserException as e:
        logger.error("Error parsing output", error=e)
        record_event("output_parse_fallback", node="tool_calls_reply")
        response.content = AgentOutput(placeholder=raw, videos=[]).model_dump_json()
    return response


def should_continue(state: AgentState) -> Literal["end", "continue"]:
    """Determine if the agent should continue or end based on the last message.

    Args:
        state: The current agent state containing messages.

    Returns:
        Literal["end", "continue"]: "end" if there are no tool calls, "continue" otherwise.
    """
    messages = state.messages
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


def build_workflow() -> CompiledStateGraph:
    """Build the workflow for the agent.

    Returns:
        CompiledStateGraph: The compiled workflow for the agent.
    """
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            # If `tools`, then we call the tool node.
            "continue": "tools",
            # Otherwise we finish.
            "end": END,
        },
    )

    workflow.add_edge("tools", "agent")

    return workflow.compile()


react_agent = build_workflow()
