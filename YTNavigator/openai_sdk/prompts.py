"""The prompts, hand-written -- this file is the prompt-engineering content.

The three templates are the original app's `app/services/agent/prompts.py`
verbatim (typos included: the words are the experiment), with the langchain
machinery around them replaced by hand-written equivalents:

- `PydanticOutputParser.get_format_instructions()` becomes
  `format_instructions()`: langchain's own preamble text with the JSON schema
  spelled out by hand. The schema field descriptions are the original pydantic
  `Field(description=...)` strings.
- `AgentGraph._pretty_str_tools()` becomes `pretty_str_tools()`, same layout.

So the model reads the same words on this side as on the LangGraph side, and
the same schema byLLM derives from its `obj` + `sem` declarations.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

# --- the three system prompts, verbatim from the original app -----------------

SYSTEM_PROMPT_TEMPLATE = """**Name:** YTNavigator
**Role:** Expert YouTube Assistant | Friendly AI Guide

**# TASK:**
Please provide comprehensive, relevant information about the channel below.

**# INSTRUCTIONS:**
- **Use Tools Wisely:** Kindly access tools only when necessary to answer requests requiring video information.
- **Detailed Responses:** Please offer thorough explanations and guidance.
- **Channel Queries:** We would appreciate if you use provided channel data to answer questions.
- **Tool Usage:** Thank you for ensuring proper use of tools,Always use the tools to get the latest data.

**# CHANNEL DATA:**
{channel}

**# USER DATA:**
{user}

**# FINAL ANSWER FORMAT:**
{format_instructions}

Please respond with only a user friendly object that respects the above format instructions,no other text or comments.

**# IMPORTANT:**
- **Format Compliance:** Please adhere to format instructions; we appreciate your attention to detail.
- **Accuracy:** We kindly ask you to avoid hallucinations; please rely solely on tool data or the provided channel data.
- **Tone:** Adopt a tone that is related to the channel content, please address the user directly.

**# NOTE:**
We would greatly appreciate your adherence to format instructions to ensure valid responses.
"""

ROUTE_QUERY_SYSTEM_PROMPT = """**# ROUTING ASSISTANT:**
You are an intelligent routing assistant that determines whether a user's message requires external information from available tools.

**# YOUR TASK:**
Please analyze the user's message and decide if it requires information from the provided tools.

**# DECISION GUIDELINES:**
- **Respond with "Yes" if:**
  - The message asks about specific channel content or channel-related information
  - The query might benefit from tool-based information retrieval
  - There's any uncertainty about whether tools might help answer the query
  - The answer might be concluded from the database or fetching a list of videos.
  Examples:
    - "What is the main channel topic?"
    - "List the videos about [topic]"
    - "Explain [concept]"
    - "How many videos are there in the channel?"

- **Respond with "No" if:**
  - The message is a simple greeting or conversational exchange
  - The query can be answered without additional information sources
  - The message requires only general knowledge or conversation

- **Respond with "Not relevant" if:**
  - The message is unrelated to the channel or available tools
  - The query falls outside the scope of your capabilities
  - The user is asking about your technical details as an LLM.

If you felt unsure about the answer,respond with "Yes".

Channel:
```{channel}```

Tools:
```{tools}```

Format instructions:
```{format_instructions}```

Please respond with only the object that respects the the format instructions,no other text or comments."""

NON_TOOL_CALLS_SYSTEM_PROMPT = """**Name:** YTNavigator
**Role:** Expert YouTube Assistant | Friendly AI Guide

**# TASK:**
You're a helpful assistant that can answer questions about the given channel.

**# INSTRUCTIONS:**
- **Try not mentioning the channel Id if not necessary.**
- **Be friendly and professional with the user.**
- **Always try to answer the question based on the channel data.**
- **If you are unsure about the answer,just say "I don't know".**

Channel:
```{channel}```

User:
```{user}```"""

# The "Not relevant" route never reaches a model, on any side.
STATIC_REPLY = "I'm sorry, I can't answer, please try again with a different question."


# --- format instructions ------------------------------------------------------

# langchain's PYDANTIC_FORMAT_INSTRUCTIONS preamble, verbatim: the original
# router and agent prompts carried these exact sentences via
# `PydanticOutputParser.get_format_instructions()`.
_FORMAT_PREAMBLE = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```"""

# AgentRouterOutput: answer: Literal["Yes", "No", "Not relevant"].
ROUTER_SCHEMA: dict[str, Any] = {
    "properties": {
        "answer": {"enum": ["Yes", "No", "Not relevant"], "title": "Answer", "type": "string"}
    },
    "required": ["answer"],
}

# AgentOutput / AgentOutputVideos / AgentOutputTimestamp, descriptions verbatim
# from the original `app/schemas/agent.py` Field(...) strings.
ANSWER_SCHEMA: dict[str, Any] = {
    "$defs": {
        "AgentOutputTimestamp": {
            "description": "The timestamp of the agent output.",
            "properties": {
                "start": {"description": "Start time of the segment", "title": "Start", "type": "string"},
                "end": {"description": "End time of the segment", "title": "End", "type": "string"},
                "description": {
                    "description": "Description of the segment (why it's relevant)",
                    "title": "Description",
                    "type": "string",
                },
            },
            "required": ["start", "end", "description"],
            "title": "AgentOutputTimestamp",
            "type": "object",
        },
        "AgentOutputVideos": {
            "description": "The output videos of the agent.",
            "properties": {
                "title": {"description": "Title of the video", "title": "Title", "type": "string"},
                "id": {"description": "Id of the video example: vxKimq_y0N5", "title": "Id", "type": "string"},
                "timestamps": {
                    "description": "List of timestamps where you found the related information",
                    "items": {"$ref": "#/$defs/AgentOutputTimestamp"},
                    "title": "Timestamps",
                    "type": "array",
                },
                "description": {
                    "description": "Description of the video related to the conversation",
                    "title": "Description",
                    "type": "string",
                },
                "thumbnail_url": {
                    "description": "Real Youtube Thumbnail url of the video from the tool results",
                    "title": "Thumbnail Url",
                    "type": "string",
                },
            },
            "required": ["title", "id", "timestamps", "description", "thumbnail_url"],
            "title": "AgentOutputVideos",
            "type": "object",
        },
    },
    "description": "The output of the agent.",
    "properties": {
        "placeholder": {
            "description": "A user-friendly message that answers the user's request based on the results found.",
            "title": "Placeholder",
            "type": "string",
        },
        "videos": {
            "description": "List of relevant videos that address the user's query,Don't Hallucinate videos,"
            "limited to actual videos in the channel from the tool results.",
            "items": {"$ref": "#/$defs/AgentOutputVideos"},
            "title": "Videos",
            "type": "array",
        },
    },
    "required": ["placeholder", "videos"],
    "title": "AgentOutput",
    "type": "object",
}


def format_instructions(schema: Mapping[str, Any]) -> str:
    """What `PydanticOutputParser.get_format_instructions()` was producing."""
    return _FORMAT_PREAMBLE.format(schema=json.dumps(schema))


def pretty_str_tools(tools: Sequence[Mapping[str, Any]]) -> str:
    """The router prompt's tool listing -- `AgentGraph._pretty_str_tools`, same
    layout, over openai function specs instead of StructuredTools."""
    string_wrapper = ""
    for idx, tool in enumerate(tools):
        fn = tool["function"]
        string_wrapper += f"""## Tool {idx + 1}:
            * Name: {fn["name"]}
            * Description: {fn["description"]}
            """
    return string_wrapper


__all__ = [
    "ANSWER_SCHEMA",
    "NON_TOOL_CALLS_SYSTEM_PROMPT",
    "ROUTER_SCHEMA",
    "ROUTE_QUERY_SYSTEM_PROMPT",
    "STATIC_REPLY",
    "SYSTEM_PROMPT_TEMPLATE",
    "format_instructions",
    "pretty_str_tools",
]
