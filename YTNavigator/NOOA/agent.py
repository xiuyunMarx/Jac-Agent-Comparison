"""The YT-Navigator chat agent in NVIDIA Object-Oriented Agents (NOOA).

Same function as ../byLLM/nodes.jac, written the way NOOA's docs say to write
an agent: no graph, no walker. One class, `YTNavigator(Agent)`.

    Jac                                            NOOA
    -------------------------------------------    ------------------------------------------
    Router node + `visit [-->] by llm(select=1)`   route(...) -> RouteDecision    Predict
      (candidates described by node `sem`s)          (the three node sems are the docstring)
    def direct_reply(...) -> AgentAnswer by llm    direct_reply(...) -> AgentAnswer  Predict
    def tool_reply(...) by llm(tools=[...],        tool_reply(...) -> AgentAnswer    CodeAct,
        max_react_iterations=6)                      max_iterations=6; the two tools are
                                                     visible methods the generated code calls
    similarity_videos_search / execute_query       regular methods on the agent (call-logged)
    StaticReply node                               the constant STATIC_REPLY, no model call
    walker ChatAgent (route -> reply, fallbacks)   chat(), a hidden deterministic method

An async method ending in `...` is implemented by the LLM: its docstring is the
prompt, `Annotated[..., "..."]` parameter descriptions and `Field(description=...)`
are the `sem` strings, the return annotation is the validated contract.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field

from nooa import Agent, EventQuery, hidden, strategy
from nooa.config import CodeActConfig, PredictConfig
from nooa.config.truncation_config import FormatConfig, TruncationConfig
from nooa.strategies import CodeActStrategy, PredictStrategy
from nooa.unifiedllm import CompletionClient, LLMResponse

# ---------------------------------------------------------------------------
# Model: the same knobs as nodes.jac. $BENCH_MODEL (bare id, default glm-5.2)
# in litellm's shape on the OpenAI-compatible endpoint from $OPENAI_BASE_URL.
# ---------------------------------------------------------------------------

BENCH_MODEL: str = os.environ.get("BENCH_MODEL", "glm-5.2")
MODEL_NAME: str = BENCH_MODEL if "/" in BENCH_MODEL else f"openai/{BENCH_MODEL}"
MODEL_BASE: str = os.environ.get("OPENAI_BASE_URL", "https://ollama.com/v1")
API_KEY: Annotated[str, hidden] = os.environ.get("OPENAI_API_KEY", "")
# Pinned identically on every side, or the benchmark measures the model
# rather than the framework. gpt-5 / o-series reject any temperature but the
# default; litellm's drop_params removes it there.
TEMPERATURE: float = 0.0
REASONING_MODEL: bool = BENCH_MODEL.split("/")[-1].startswith(("gpt-5", "o1", "o3", "o4"))
# byLLM's `max_react_iterations=6`: model turns in one tool_reply.
MAX_REACT_ITERATIONS: int = 6

STATIC_REPLY: str = "I'm sorry, I can't answer, please try again with a different question."
PARSE_FAILURE_REPLY: str = "I'm sorry, I couldn't generate a valid answer."
# byLLM's walker default -- the synthetic user the benchmark creates.
USER_INFO: str = "username: benchmark , email: benchmark@localhost"


# ---------------------------------------------------------------------------
# Retrieval plumbing: ../byLLM/retrieval.py loaded from the sibling, as the
# openai_sdk arm does, so every arm hits one copy against one set of tables.
# Hidden from generated code: the model reaches the database only through
# the two logged tool methods below.
# ---------------------------------------------------------------------------

_RETRIEVAL_PATH = Path(__file__).resolve().parent.parent / "byLLM" / "retrieval.py"


def _load_retrieval() -> Any:
    if "ytnav_retrieval" in sys.modules:
        return sys.modules["ytnav_retrieval"]
    spec = importlib.util.spec_from_file_location("ytnav_retrieval", _RETRIEVAL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Shared retrieval module not found: {_RETRIEVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ytnav_retrieval"] = module
    spec.loader.exec_module(module)
    return module


with hidden:
    retrieval = _load_retrieval()


# ---------------------------------------------------------------------------
# Typed results: the Jac `obj` declarations, field for field; `sem` strings
# are Field descriptions, rendered into the schema the model sees.
# ---------------------------------------------------------------------------


class AnswerTimestamp(BaseModel):
    start: str = Field("", description="Start time of the segment.")
    end: str = Field("", description="End time of the segment.")
    description: str = Field("", description="Why this segment is relevant.")


class AnswerVideo(BaseModel):
    id: str = Field("", description="Id of the video, exactly as returned by the tools (example: vxKimq_y0N5).")
    title: str = Field("", description="Title of the video.")
    description: str = Field("", description="Description of the video related to the conversation.")
    thumbnail_url: str = Field("", description="Real YouTube thumbnail url of the video from the tool results.")
    timestamps: list[AnswerTimestamp] = Field(
        default_factory=list, description="Timestamps where the related information was found."
    )


class AgentAnswer(BaseModel):
    placeholder: str = Field(
        "", description="A user-friendly message that answers the user's request based on the results found."
    )
    videos: list[AnswerVideo] = Field(
        default_factory=list,
        description=(
            "The source videos for the answer. Whenever the answer draws on tool results, list the 1-5 videos "
            "whose transcript chunks or metadata support it, using their real ids, titles and thumbnail urls "
            "exactly as returned by the tools - only the videos actually used, not every search hit. Do not "
            "hallucinate videos. Empty only when no video content was used (greetings, refusals)."
        ),
    )


class Route(str, Enum):
    """Which reply handles the message. The values are the benchmark's route labels."""

    TOOL_REPLY = "Yes"
    DIRECT_REPLY = "No"
    STATIC_REPLY = "Not relevant"


class RouteDecision(BaseModel):
    """The router's choice. A named model rather than a bare enum: NOOA wraps a bare enum in a
    one-field model whose key only appears in the response_format schema, which ollama.com
    ignores for GLM; the class rendered in the prompt is what the model reads."""

    route: Route = Field(description="TOOL_REPLY ('Yes'), DIRECT_REPLY ('No') or STATIC_REPLY ('Not relevant').")


@dataclass
class ChatResult:
    """What one question produced -- the fields the runner's record needs."""

    route: str = ""
    answer: AgentAnswer | None = None
    fallback_events: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The model seam: NOOA's litellm CompletionClient, subclassed only to record
# one {"model", "latency_s", "prompt_tokens", "completion_tokens"} row per
# call, the shared benchmark schema's llm_calls element. main.py slices the
# list per question, like the other arms' trackers.
# ---------------------------------------------------------------------------

_calls: list[dict[str, Any]] = []


def llm_call_count() -> int:
    return len(_calls)


def llm_calls_since(index: int) -> list[dict[str, Any]]:
    return list(_calls[index:])


class TracedClient(CompletionClient):
    async def acall(self, messages: list[dict[str, Any]], *args: Any, **kwargs: Any) -> LLMResponse:
        started = time.perf_counter()
        response: LLMResponse | None = None
        try:
            response = await super().acall(messages, *args, **kwargs)
            return response
        finally:
            usage = (response.usage or {}) if response is not None else {}
            _calls.append(
                {
                    "model": BENCH_MODEL,
                    "latency_s": round(time.perf_counter() - started, 4),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
            )


LLM_CALL_PARAMS: dict[str, Any] = {"reasoning_effort": "minimal"} if REASONING_MODEL else {}
llm = TracedClient(
    model=MODEL_NAME,
    api_base=MODEL_BASE,
    api_key=API_KEY or "ollama",
    temperature=TEMPERATURE,
    drop_params=True,       # litellm drops what a model rejects (gpt-5: temperature)
    **LLM_CALL_PARAMS,
)


def by_llm() -> PredictStrategy:
    """`by llm(temperature=0.0)`: one structured call, one correction turn."""
    return PredictStrategy(config=PredictConfig(temperature=TEMPERATURE, max_retries=2))


def by_llm_with_tools() -> CodeActStrategy:
    """`by llm(tools=[...], temperature=0.0, max_react_iterations=6)`: the tool loop.

    The model acts by writing Python that calls the visible tool methods on
    `self`, and finishes with `return_result(AgentAnswer(...))`.
    """
    return CodeActStrategy(config=CodeActConfig(temperature=TEMPERATURE, max_iterations=MAX_REACT_ITERATIONS))


# Arguments are rendered whole (channel data, conversation, tool output).
UNTRUNCATED = TruncationConfig(prefill_format=FormatConfig(max_string=None, max_length=None, max_depth=None))


# ---------------------------------------------------------------------------
# The agent.
# ---------------------------------------------------------------------------


class YTNavigator(Agent, llm=llm, event_query=EventQuery.current_call(), truncation=UNTRUNCATED):
    """You are YTNavigator, an expert YouTube assistant and friendly AI guide for one channel."""

    def __init__(self, channel_id: str, **kwargs: Any):
        super().__init__(**kwargs)
        self._channel_id = channel_id
        self._tool_calls: list[dict[str, Any]] = []

    # -- tools: regular methods, the capability surface generated code calls --

    def similarity_videos_search(
        self, query: Annotated[str, "Natural-language search query to find similar transcript content."]
    ) -> str:
        """Advanced semantic video search tool powered by vector embeddings. Use this tool to: find
        videos that match the semantic meaning of your query, not just exact keywords; discover
        relevant video content across different topics and contexts; retrieve video chunks that
        closely align with the intent of your search; get an idea about the channel's content.
        Ideal for complex information retrieval tasks over video transcripts. The tool may return
        some irrelevant results; provide the user with the most relevant ones."""
        self._tool_calls.append({"name": "similarity_videos_search", "args": {"query": query}})
        try:
            return str(retrieval.search_videos_impl(query, self._channel_id))
        except Exception as e:  # noqa: BLE001 - the model reads the failure
            return f"Search failed: {e}"

    def execute_query(
        self, query: Annotated[str, "A single SELECT statement against app_video and/or app_videochunk."]
    ) -> str:
        """Powerful SQL query execution tool (PostgreSQL syntax, SELECT only) for advanced data
        retrieval and analysis over the channel's videos: joining the app_video and app_videochunk
        tables, filtering and aggregating video metadata, counting videos, or retrieving specific
        subsets of data. app_video columns: id, title, thumbnail, published_at, channel_id.
        app_videochunk columns: id, video_id, start, end, text."""
        self._tool_calls.append({"name": "execute_query", "args": {"query": query}})
        try:
            return str(retrieval.run_sql_impl(query))
        except Exception as e:  # noqa: BLE001
            return f"Query failed: {e}"

    # -- the three judgments ---------------------------------------------------

    @strategy(by_llm())
    async def route(
        self,
        channel: Annotated[str, "The channel data."],
        conversation: Annotated[str, "Recent conversation turns, oldest first (may be empty)."],
        message: Annotated[str, "The user message to route."],
    ) -> RouteDecision:
        """You are an intelligent routing assistant for a YouTube-channel Q&A agent. Analyze the
        user's message and decide which reply handles it:

        - TOOL_REPLY ("Yes"): messages that ask about specific channel content or channel-related
          information, might benefit from tool-based information retrieval (searching transcripts,
          listing or counting videos), or where there is any uncertainty about whether tools might
          help. Examples: 'What is the main channel topic?', 'List the videos about X', 'Explain X',
          'How many videos are there in the channel?'. When unsure, choose this.
        - DIRECT_REPLY ("No"): simple greetings or conversational exchanges that can be answered
          without additional information sources, using only general knowledge or the provided
          channel data.
        - STATIC_REPLY ("Not relevant"): messages that are unrelated to the channel or available
          tools, fall outside the assistant's scope, or ask about the assistant's technical details
          as an LLM; they get a fixed apology.

        If unsure, choose TOOL_REPLY."""
        ...

    @strategy(by_llm())
    async def direct_reply(
        self,
        channel: Annotated[str, "The channel data to ground the answer in."],
        user: Annotated[str, "Information about the user being addressed."],
        conversation: Annotated[str, "Recent conversation turns, oldest first (may be empty)."],
        message: Annotated[str, "The user's latest message to answer."],
    ) -> AgentAnswer:
        """Answer the user's message based on the given channel data. Avoid mentioning the channel
        id unless necessary. Be friendly and professional. If unsure about the answer, just say
        'I don't know'. Do not invent videos; leave videos empty for conversational replies."""
        ...

    @strategy(by_llm_with_tools())
    async def tool_reply(
        self,
        channel: Annotated[str, "The channel data the assistant serves."],
        user: Annotated[str, "Information about the user being addressed."],
        conversation: Annotated[str, "Recent conversation turns, oldest first (may be empty)."],
        message: Annotated[str, "The user's latest request to fulfil using the tools."],
    ) -> AgentAnswer:
        """Provide comprehensive, relevant information about the channel below, using the tools
        (self.similarity_videos_search, self.execute_query) to gather video information before
        answering. Always use the tools to get the latest data; rely solely on tool data or the
        provided channel data - avoid hallucinations. Offer thorough explanations, adopt a tone
        related to the channel content, and address the user directly. Cite the relevant videos
        with their ids, titles, thumbnails, and timestamps from the tool results."""
        ...

    # -- the walk: Python decides, with the same fallback ladder as the walker --

    @hidden
    async def chat(self, channel: str, conversation: str, message: str, result: ChatResult | None = None) -> ChatResult:
        """One question: route, then the route's reply. Pass a ChatResult to keep what was
        decided before a failure, as byLLM's walker keeps its route and fallback events."""
        result = result if result is not None else ChatResult()
        self._tool_calls = result.tool_calls

        try:
            route = (await self.route(channel, conversation, message)).route
        except Exception as e:  # noqa: BLE001 - the run must survive a bad route
            # No valid choice: mirror the walker's fallback and take the direct reply.
            result.fallback_events.append({"event": "router_error_fallback", "error": str(e)})
            route = Route.DIRECT_REPLY
        result.route = route.value

        if route == Route.STATIC_REPLY:
            result.answer = AgentAnswer(placeholder=STATIC_REPLY, videos=[])
            return result

        reply = self.tool_reply if route == Route.TOOL_REPLY else self.direct_reply
        try:
            result.answer = await reply(channel, USER_INFO, conversation, message)
        except Exception as e:  # noqa: BLE001 - a malformed answer costs one question, not the run
            result.fallback_events.append({"event": "output_parse_fallback", "node": reply.__name__, "error": str(e)})
            result.answer = AgentAnswer(placeholder=PARSE_FAILURE_REPLY, videos=[])
        return result


__all__ = [
    "MAX_REACT_ITERATIONS",
    "STATIC_REPLY",
    "USER_INFO",
    "AgentAnswer",
    "AnswerTimestamp",
    "AnswerVideo",
    "ChatResult",
    "Route",
    "YTNavigator",
    "llm",
    "llm_call_count",
    "llm_calls_since",
    "retrieval",
]
