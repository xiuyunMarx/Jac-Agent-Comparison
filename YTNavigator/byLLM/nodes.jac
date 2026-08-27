"""The byLLM counterpart of YT-Navigator's LangGraph agent.

Mirrors the original graph topology as an object-spatial graph:

    root -> Router -> { StaticReply | DirectReply | ToolReply }

The router is byLLM's visit-by routing: at the Router node the walker runs
`visit [-->] by llm(...)` and the model picks exactly one of the three reply
nodes. The candidates are described to the model by the reply nodes' `sem`
strings (the same three-way guidelines as ROUTE_QUERY_SYSTEM_PROMPT), the
`intent` carries the router instructions, and `incl_info` passes the channel
data, conversation and message. The direct/tool replies produce the same
structured answer shape as AgentOutput, and the tool reply runs byLLM's
built-in ReAct loop over the same two tools. All calls use the single `llm`
model at temperature 0.
"""

import os;
import from jaclang.byllm.lib { Model }
import from tools { similarity_videos_search, execute_query }

# $BENCH_MODEL (bare id, default glm-5.2) in litellm's shape, on the OpenAI-compatible
# endpoint from $OPENAI_BASE_URL. byLLM 0.31.1 drops config["base_url"] on the litellm
# path, so the endpoint reaches litellm through OPENAI_API_BASE instead.
glob BENCH_MODEL: str = os.environ.get("BENCH_MODEL", "glm-5.2");
glob _base_set: bool = (os.environ.setdefault(
    "OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", "https://ollama.com/v1")
) is not None);
glob llm = Model(
    model_name=(BENCH_MODEL if "/" in BENCH_MODEL else f"openai/{BENCH_MODEL}"),
    api_key=os.environ.get("OPENAI_API_KEY", "")
);

glob STATIC_REPLY: str = "I'm sorry, I can't answer, please try again with a different question.";

# Goal text for the visit-by routing call (the original ROUTE_QUERY_SYSTEM_PROMPT).
glob ROUTE_INTENT: str = "You are an intelligent routing assistant for a YouTube-channel Q&A agent. Analyze the user's message and decide which reply node should handle it: ToolReply when answering requires external information from the agent's tools (semantic transcript search and SQL over video metadata), DirectReply for greetings or conversational exchanges answerable from general knowledge or the provided channel data, StaticReply when the message is unrelated to the channel or asks about the assistant's technical details as an LLM. Examples that need tools: 'What is the main channel topic?', 'List the videos about X', 'Explain X', 'How many videos are there in the channel?'. If unsure, choose ToolReply.";


obj AnswerTimestamp {
    has start: str = "";
    has end: str = "";
    has description: str = "";
}
sem AnswerTimestamp.start = "Start time of the segment.";
sem AnswerTimestamp.end = "End time of the segment.";
sem AnswerTimestamp.description = "Why this segment is relevant.";

obj AnswerVideo {
    has id: str = "";
    has title: str = "";
    has description: str = "";
    has thumbnail_url: str = "";
    has timestamps: list[AnswerTimestamp] = [];
}
sem AnswerVideo.id = "Id of the video, exactly as returned by the tools (example: vxKimq_y0N5).";
sem AnswerVideo.title = "Title of the video.";
sem AnswerVideo.description = "Description of the video related to the conversation.";
sem AnswerVideo.thumbnail_url = "Real YouTube thumbnail url of the video from the tool results.";
sem AnswerVideo.timestamps = "Timestamps where the related information was found.";

obj AgentAnswer {
    has placeholder: str = "";
    has videos: list[AnswerVideo] = [];
}
sem AgentAnswer.placeholder = "A user-friendly message that answers the user's request based on the results found.";
sem AgentAnswer.videos = "The source videos for the answer. Whenever the answer draws on tool results, list the 1-5 videos whose transcript chunks or metadata support it, using their real ids, titles and thumbnail urls exactly as returned by the tools - only the videos actually used, not every search hit. Do not hallucinate videos. Empty only when no video content was used (greetings, refusals).";


def direct_reply(channel: str, user: str, conversation: str, message: str) -> AgentAnswer by llm(temperature=0.0);
sem direct_reply = "You are YTNavigator, an expert YouTube assistant and friendly AI guide. Answer the user's message based on the given channel data. Avoid mentioning the channel id unless necessary. Be friendly and professional. If unsure about the answer, just say 'I don't know'. Do not invent videos; leave videos empty for conversational replies.";
sem direct_reply.channel = "The channel data to ground the answer in.";
sem direct_reply.user = "Information about the user being addressed.";
sem direct_reply.conversation = "Recent conversation turns, oldest first (may be empty).";
sem direct_reply.message = "The user's latest message to answer.";

def tool_reply(channel: str, user: str, conversation: str, message: str) -> AgentAnswer by llm(
    tools=[similarity_videos_search, execute_query],
    temperature=0.0,
    max_react_iterations=6
);
sem tool_reply = "You are YTNavigator, an expert YouTube assistant and friendly AI guide. Provide comprehensive, relevant information about the channel below, using the tools to gather video information before answering. Always use the tools to get the latest data; rely solely on tool data or the provided channel data - avoid hallucinations. Offer thorough explanations, adopt a tone related to the channel content, and address the user directly. Cite the relevant videos with their ids, titles, thumbnails, and timestamps from the tool results.";
sem tool_reply.channel = "The channel data the assistant serves.";
sem tool_reply.user = "Information about the user being addressed.";
sem tool_reply.conversation = "Recent conversation turns, oldest first (may be empty).";
sem tool_reply.message = "The user's latest request to fulfil using the tools.";


def coerce_answer(result: object) -> AgentAnswer {
    # byLLM occasionally hands back a plain string or dict instead of the
    # declared AgentAnswer (e.g. when the model's final message escapes the
    # output converter); normalize so downstream code can rely on the type.
    if isinstance(result, AgentAnswer) {
        return result;
    }
    if isinstance(result, dict) {
        videos = [];
        for raw_video in (result.get("videos") or []) {
            if isinstance(raw_video, dict) {
                timestamps = [];
                for raw_ts in (raw_video.get("timestamps") or []) {
                    if isinstance(raw_ts, dict) {
                        timestamps.append(AnswerTimestamp(
                            start=str(raw_ts.get("start") or ""),
                            end=str(raw_ts.get("end") or ""),
                            description=str(raw_ts.get("description") or "")
                        ));
                    }
                }
                videos.append(AnswerVideo(
                    id=str(raw_video.get("id") or ""),
                    title=str(raw_video.get("title") or ""),
                    description=str(raw_video.get("description") or ""),
                    thumbnail_url=str(raw_video.get("thumbnail_url") or ""),
                    timestamps=timestamps
                ));
            }
        }
        return AgentAnswer(placeholder=str(result.get("placeholder") or ""), videos=videos);
    }
    return AgentAnswer(placeholder=str(result), videos=[]);
}


# The reply nodes' sem strings are what the visit-by router sees for each
# candidate - they carry the three-way routing guidelines of the original
# ROUTE_QUERY_SYSTEM_PROMPT (YES -> ToolReply, NO -> DirectReply,
# NOT_RELEVANT -> StaticReply).
node Router {}
node DirectReply {}
node StaticReply {}
node ToolReply {}
sem Router = "Entry point that classifies the user's message and hands it to exactly one reply node.";
sem ToolReply = "Handles messages that ask about specific channel content or channel-related information, might benefit from tool-based information retrieval (searching transcripts, listing or counting videos), or where there is any uncertainty about whether tools might help. When unsure, choose this node.";
sem DirectReply = "Handles simple greetings or conversational exchanges that can be answered without additional information sources, using only general knowledge or the provided channel data.";
sem StaticReply = "Handles messages that are unrelated to the channel or available tools, fall outside the assistant's scope, or ask about the assistant's technical details as an LLM; replies with a fixed apology.";


walker ChatAgent {
    has message: str;
    has channel_info: str;
    has conversation: str = "";
    has user_info: str = "username: benchmark , email: benchmark@localhost";
    has route: str = "";
    has answer: AgentAnswer | None = None;
    has fallback_events: list[dict] = [];

    can start with Root entry {
        if not [-->][?:Router] {
            router_node = Router();
            here ++> router_node;
            router_node ++> DirectReply();
            router_node ++> StaticReply();
            router_node ++> ToolReply();
        }
        visit [-->][?:Router];
    }

    can decide with Router entry {
        try {
            visit [-->] by llm(
                select=1,
                temperature=0.0,
                intent=ROUTE_INTENT,
                incl_info={
                    "Channel data": self.channel_info,
                    "Recent conversation (oldest first, may be empty)": self.conversation,
                    "User message to route": self.message
                }
            ) else {
                # The model returned no valid handle: mirror the original
                # router_condition fallback and take the direct reply.
                self.fallback_events.append({"event": "router_empty_fallback"});
                visit [-->][?:DirectReply];
            }
        } except Exception as e {
            self.fallback_events.append({"event": "router_error_fallback", "error": str(e)});
            visit [-->][?:DirectReply];
        }
    }

    can reply_static with StaticReply entry {
        self.route = "Not relevant";
        self.answer = AgentAnswer(placeholder=STATIC_REPLY, videos=[]);
    }

    can reply_direct with DirectReply entry {
        self.route = "No";
        try {
            result = direct_reply(
                channel=self.channel_info,
                user=self.user_info,
                conversation=self.conversation,
                message=self.message
            );
            if not isinstance(result, AgentAnswer) {
                self.fallback_events.append({"event": "output_type_fallback", "node": "direct_reply", "got": str(type(result))});
            }
            self.answer = coerce_answer(result);
        } except Exception as e {
            self.fallback_events.append({"event": "output_parse_fallback", "node": "direct_reply", "error": str(e)});
            raw = str(getattr(e, "raw_output", "") or "");
            self.answer = AgentAnswer(placeholder=(raw if raw else "I'm sorry, I couldn't generate a valid answer."), videos=[]);
        }
    }

    can reply_tools with ToolReply entry {
        self.route = "Yes";
        try {
            result = tool_reply(
                channel=self.channel_info,
                user=self.user_info,
                conversation=self.conversation,
                message=self.message
            );
            if not isinstance(result, AgentAnswer) {
                self.fallback_events.append({"event": "output_type_fallback", "node": "tool_reply", "got": str(type(result))});
            }
            self.answer = coerce_answer(result);
        } except Exception as e {
            self.fallback_events.append({"event": "output_parse_fallback", "node": "tool_reply", "error": str(e)});
            raw = str(getattr(e, "raw_output", "") or "");
            self.answer = AgentAnswer(placeholder=(raw if raw else "I'm sorry, I couldn't generate a valid answer."), videos=[]);
        }
    }
}
