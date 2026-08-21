"""The agent: router -> {static, direct, tool} reply, written out.

The topology every side shares:

    message -> route_query -> "Not relevant" -> STATIC_REPLY
                           -> "No"           -> direct_reply   (INSTANT_LLM)
                           -> "Yes"          -> tool_reply     (POWERFUL_LLM + ReAct)

The original expresses it as a StateGraph with three conditional edges; byLLM
as an object-spatial graph a walker traverses; here it is one `if` in
`chat()`. The ReAct loop the original gets from a compiled subgraph and byLLM
from `by llm(tools=[...], max_react_iterations=6)` is ~40 lines of `while
True:` in `tool_reply`.

Structured output is prompt + parse, as the original did it: the prompts carry
the JSON schema (prompts.format_instructions) and `parse_answer` reads the
reply back, with the same recovery ladder recorded as fallback_events --
`router_parse_fallback` (keyword rescue, original naming),
`router_error_fallback` (route defaults to "No", byLLM naming),
`output_parse_fallback` (raw text becomes the placeholder, both sides'
naming; main.py counts it against answer_parsed exactly as byLLM does).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import llm
import prompts
import tools

# byLLM's `max_react_iterations=6`: model turns in one tool_reply, counting
# the final no-tools turn.
MAX_REACT_ITERATIONS = 6

ROUTES = ("Yes", "No", "Not relevant")

# byLLM's walker default -- the synthetic user benchmark_run creates.
USER_INFO = "username: benchmark , email: benchmark@localhost"

PARSE_FAILURE_REPLY = "I'm sorry, I couldn't generate a valid answer."

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


@dataclass
class AnswerTimestamp:
    start: str = ""
    end: str = ""
    description: str = ""


@dataclass
class AnswerVideo:
    id: str = ""
    title: str = ""
    description: str = ""
    thumbnail_url: str = ""
    timestamps: list[AnswerTimestamp] = field(default_factory=list)


@dataclass
class AgentAnswer:
    placeholder: str = ""
    videos: list[AnswerVideo] = field(default_factory=list)


@dataclass
class ChatResult:
    """What one question produced -- the fields main.py's record needs."""

    route: str = ""
    answer: AgentAnswer | None = None
    fallback_events: list[dict[str, Any]] = field(default_factory=list)


class AnswerParseError(ValueError):
    """The reply was not the schema'd JSON object; carries the raw text."""

    def __init__(self, message: str, raw_output: str):
        super().__init__(message)
        self.raw_output = raw_output


# --- parsing ------------------------------------------------------------------


def extract_json(text: str) -> Any:
    """The JSON value in a model reply, tolerating code fences and prose.

    PydanticOutputParser's lenient pass did the same job on the original side:
    try the text as-is, then the outermost {...} slice.
    """
    candidate = _FENCE_RE.sub("", text.strip())
    try:
        return json.loads(candidate)
    except ValueError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        return json.loads(candidate[start : end + 1])
    raise ValueError("no JSON object found")


def parse_answer(text: str) -> AgentAnswer:
    """The reply as an AgentAnswer, or AnswerParseError.

    The coercions mirror the original AgentOutput validators and byLLM's
    coerce_answer: missing/null fields become "" or [], and the video list is
    capped at 5 (the original's limit_videos_length).
    """
    if not isinstance(text, str) or not text.strip():
        raise AnswerParseError("empty reply", raw_output=text or "")
    try:
        data = extract_json(text)
    except ValueError as e:
        raise AnswerParseError(f"invalid JSON: {e}", raw_output=text) from e
    if not isinstance(data, dict):
        raise AnswerParseError(f"expected a JSON object, got {type(data).__name__}", raw_output=text)

    videos = []
    raw_videos = data.get("videos") or []
    if not isinstance(raw_videos, list):
        raw_videos = []
    for raw_video in raw_videos[:5]:
        if not isinstance(raw_video, dict):
            continue
        timestamps = []
        for raw_ts in raw_video.get("timestamps") or []:
            if isinstance(raw_ts, dict):
                timestamps.append(
                    AnswerTimestamp(
                        start=str(raw_ts.get("start") or ""),
                        end=str(raw_ts.get("end") or ""),
                        description=str(raw_ts.get("description") or ""),
                    )
                )
        videos.append(
            AnswerVideo(
                id=str(raw_video.get("id") or ""),
                title=str(raw_video.get("title") or ""),
                description=str(raw_video.get("description") or ""),
                thumbnail_url=str(raw_video.get("thumbnail_url") or ""),
                timestamps=timestamps,
            )
        )
    return AgentAnswer(placeholder=str(data.get("placeholder") or ""), videos=videos)


# --- conversation plumbing ----------------------------------------------------


def history_turns(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prior exchanges as real chat turns, oldest first.

    The last 3 exchanges, byLLM's plain-text stand-in for the original's
    1000-token `trim_messages` budget -- but sent as user/assistant messages,
    which is the layout the original put on the wire.
    """
    turns: list[dict[str, Any]] = []
    for exchange in history[-3:]:
        turns.append({"role": "user", "content": str(exchange["user"])})
        turns.append({"role": "assistant", "content": str(exchange.get("assistant") or "")})
    return turns


def _messages(system: str, history: list[dict[str, Any]], message: str) -> list[dict[str, Any]]:
    return [{"role": "system", "content": system}, *history_turns(history), {"role": "user", "content": message}]


# --- the three nodes ----------------------------------------------------------


def route_query(
    channel_info: str,
    history: list[dict[str, Any]],
    message: str,
    events: list[dict[str, Any]],
    *,
    client: Any = None,
) -> str:
    """The three-way routing decision, with the original's recovery ladder."""
    system = prompts.ROUTE_QUERY_SYSTEM_PROMPT.format(
        channel=channel_info,
        tools=prompts.pretty_str_tools(tools.TOOL_SPECS),
        format_instructions=prompts.format_instructions(prompts.ROUTER_SCHEMA),
    )
    try:
        reply = llm.complete(
            _messages(system, history, message), model=llm.instant_model(), client=client
        )
        content = reply.content or ""
        try:
            data = extract_json(content)
            answer = data.get("answer") if isinstance(data, dict) else None
            if answer in ROUTES:
                return answer
            raise ValueError(f"answer not in {ROUTES}: {answer!r}")
        except ValueError:
            # The original's OutputParserException path: scan for a routing
            # keyword, most specific first ("Not relevant" contains neither of
            # the others, but "Yes"/"No" could appear inside prose).
            for route in ["Not relevant", "Yes", "No"]:
                if route in content:
                    events.append({"event": "router_parse_fallback", "recovered_route": route})
                    return route
            raise ValueError(f"unroutable reply: {content[:200]!r}")
    except Exception as e:  # noqa: BLE001 - the run must survive a bad route
        # The original router_condition error fallback: default to a direct
        # reply, but make the recovery visible to the benchmark (byLLM's
        # event name).
        events.append({"event": "router_error_fallback", "error": str(e)})
        return "No"


def _finish(
    node: str, content: str, events: list[dict[str, Any]]
) -> AgentAnswer:
    """Parse a reply node's final text, falling back exactly like both sides:
    the raw text becomes the placeholder and the recovery is recorded."""
    try:
        return parse_answer(content)
    except AnswerParseError as e:
        events.append({"event": "output_parse_fallback", "node": node, "error": str(e)})
        raw = (e.raw_output or "").strip()
        return AgentAnswer(placeholder=raw if raw else PARSE_FAILURE_REPLY, videos=[])


def direct_reply(
    channel_info: str,
    history: list[dict[str, Any]],
    message: str,
    events: list[dict[str, Any]],
    *,
    client: Any = None,
) -> AgentAnswer:
    """The "No" route: answer from channel data alone, on the instant model."""
    system = (
        prompts.NON_TOOL_CALLS_SYSTEM_PROMPT.format(channel=channel_info, user=USER_INFO)
        + "\n\n**# FINAL ANSWER FORMAT:**\n"
        + prompts.format_instructions(prompts.ANSWER_SCHEMA)
        + "\n\nPlease respond with only a user friendly object that respects the above format "
        "instructions,no other text or comments."
    )
    reply = llm.complete(_messages(system, history, message), model=llm.instant_model(), client=client)
    return _finish("direct_reply", reply.content or "", events)


def tool_reply(
    channel_info: str,
    history: list[dict[str, Any]],
    message: str,
    events: list[dict[str, Any]],
    *,
    client: Any = None,
) -> AgentAnswer:
    """The "Yes" route: the ReAct loop, written out.

    What `by llm(tools=[...], max_react_iterations=6)` and the compiled
    react_graph provide elsewhere: call the model with the tool specs; run
    every requested call in order (a `for` loop is already serial, where
    byLLM needs `mark_serialize` and LangGraph a custom tool node to get
    that); loop until a turn arrives without tool calls, which is the final
    answer. Every requested call is answered before the next turn -- a
    conversation with an unanswered tool call is invalid for the provider --
    and the last turn the cap allows is asked without the tool specs, so the
    loop always ends with an answer rather than mid-thought.
    """
    system = prompts.SYSTEM_PROMPT_TEMPLATE.format(
        channel=channel_info,
        user=USER_INFO,
        format_instructions=prompts.format_instructions(prompts.ANSWER_SCHEMA),
    )
    messages = _messages(system, history, message)
    model = llm.powerful_model()
    iterations = 0

    while True:
        at_cap = iterations >= MAX_REACT_ITERATIONS - 1
        reply = llm.complete(
            messages, model=model, tools=None if at_cap else tools.TOOL_SPECS, client=client
        )
        messages.append(llm.assistant_turn(reply))
        iterations += 1

        calls = list(getattr(reply, "tool_calls", None) or [])
        if not calls:
            return _finish("tool_reply", reply.content or "", events)

        for call in calls:
            result = tools.dispatch(call.function.name, call.function.arguments)
            messages.append(llm.tool_turn(call.id, call.function.name, result))


# --- the walk -----------------------------------------------------------------


def chat(
    message: str,
    channel_info: str,
    history: list[dict[str, Any]],
    *,
    client: Any = None,
    result: ChatResult | None = None,
) -> ChatResult:
    """One question through the graph: route, then the route's reply.

    Pass a ChatResult to keep what was decided before a failure: byLLM's
    walker retains its route and fallback_events when a reply node raises,
    and the benchmark record should carry the same partial truth here.
    """
    if result is None:
        result = ChatResult()
    result.route = route_query(channel_info, history, message, result.fallback_events, client=client)

    if result.route == "Not relevant":
        result.answer = AgentAnswer(placeholder=prompts.STATIC_REPLY, videos=[])
    elif result.route == "Yes":
        result.answer = tool_reply(channel_info, history, message, result.fallback_events, client=client)
    else:
        result.answer = direct_reply(channel_info, history, message, result.fallback_events, client=client)
    return result


__all__ = [
    "MAX_REACT_ITERATIONS",
    "ROUTES",
    "USER_INFO",
    "AgentAnswer",
    "AnswerParseError",
    "AnswerTimestamp",
    "AnswerVideo",
    "ChatResult",
    "chat",
    "direct_reply",
    "extract_json",
    "history_turns",
    "parse_answer",
    "route_query",
    "tool_reply",
]
