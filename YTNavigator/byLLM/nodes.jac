"""The byLLM counterpart of YT-Navigator's LangGraph agent.

Mirrors the original graph topology as an object-spatial graph:

    root -> Router -> { StaticReply | DirectReply | ToolReply }

The router classifies each message (same three-way decision and guidelines as
ROUTE_QUERY_SYSTEM_PROMPT), the direct/tool replies produce the same
structured answer shape as AgentOutput, and the tool reply runs byLLM's
built-in ReAct loop over the same two tools. Models match the original app:
INSTANT_LLM (default gpt-4o-mini) for routing/direct replies, POWERFUL_LLM
(default gpt-4o) for the tool agent, both at temperature 0.
"""

import os;
import from jaclang.byllm.lib { Model }
import from tools { similarity_videos_search, execute_query }

# Same INSTANT_LLM / POWERFUL_LLM values as the original app. Plain names are
# OpenAI models ("gpt-4o-mini", needs OPENAI_API_KEY); provider prefixes work
# too - langchain-style "groq:model" is normalized to litellm's "groq/model".
glob router_llm = Model(model_name=os.environ.get("INSTANT_LLM", "gpt-4o-mini").replace(":", "/"));
glob agent_llm = Model(model_name=os.environ.get("POWERFUL_LLM", "gpt-4o").replace(":", "/"));

glob STATIC_REPLY: str = "I'm sorry, I can't answer, please try again with a different question.";


enum Route { YES, NO, NOT_RELEVANT }
sem Route.YES = "The message asks about specific channel content or channel-related information, might benefit from tool-based information retrieval (searching transcripts, listing or counting videos), or there is any uncertainty about whether tools might help. When unsure, choose YES.";
sem Route.NO = "The message is a simple greeting or conversational exchange that can be answered without additional information sources, using only general knowledge or the provided channel data.";
sem Route.NOT_RELEVANT = "The message is unrelated to the channel or available tools, falls outside the assistant's scope, or asks about the assistant's technical details as an LLM.";


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


def route_query(channel: str, conversation: str, message: str) -> Route by router_llm(temperature=0.0);
sem route_query = "You are an intelligent routing assistant for a YouTube-channel Q&A agent. Analyze the user's message and decide whether answering it requires external information from the agent's tools (semantic transcript search and SQL over video metadata). Examples that need tools: 'What is the main channel topic?', 'List the videos about X', 'Explain X', 'How many videos are there in the channel?'. If unsure, answer YES.";
sem route_query.channel = "The channel data the assistant serves.";
sem route_query.conversation = "Recent conversation turns, oldest first (may be empty).";
sem route_query.message = "The user's latest message to route.";

def direct_reply(channel: str, user: str, conversation: str, message: str) -> AgentAnswer by router_llm(temperature=0.0);
sem direct_reply = "You are YTNavigator, an expert YouTube assistant and friendly AI guide. Answer the user's message based on the given channel data. Avoid mentioning the channel id unless necessary. Be friendly and professional. If unsure about the answer, just say 'I don't know'. Do not invent videos; leave videos empty for conversational replies.";
sem direct_reply.channel = "The channel data to ground the answer in.";
sem direct_reply.user = "Information about the user being addressed.";
sem direct_reply.conversation = "Recent conversation turns, oldest first (may be empty).";
sem direct_reply.message = "The user's latest message to answer.";

def tool_reply(channel: str, user: str, conversation: str, message: str) -> AgentAnswer by agent_llm(
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


def route_to_string(decision: Route) -> str {
    if decision == Route.YES {
        return "Yes";
    } elif decision == Route.NOT_RELEVANT {
        return "Not relevant";
    }
    return "No";
}


node Router {}
node DirectReply {}
node StaticReply {}
node ToolReply {}


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
            decision = route_query(
                channel=self.channel_info,
                conversation=self.conversation,
                message=self.message
            );
            self.route = route_to_string(decision);
        } except Exception as e {
            # Mirrors the original router_condition error fallback: default to
            # a direct reply, but make the recovery visible to the benchmark.
            self.fallback_events.append({"event": "router_error_fallback", "error": str(e)});
            self.route = "No";
        }
        if self.route == "Yes" {
            visit [-->][?:ToolReply];
        } elif self.route == "Not relevant" {
            visit [-->][?:StaticReply];
        } else {
            visit [-->][?:DirectReply];
        }
    }

    can reply_static with StaticReply entry {
        self.answer = AgentAnswer(placeholder=STATIC_REPLY, videos=[]);
    }

    can reply_direct with DirectReply entry {
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
            # Mirrors the original parse fallback: wrap whatever text came
            # back as the answer, and record the recovery.
            self.fallback_events.append({"event": "output_parse_fallback", "node": "direct_reply", "error": str(e)});
            raw = str(getattr(e, "raw_output", "") or "");
            self.answer = AgentAnswer(placeholder=(raw if raw else "I'm sorry, I couldn't generate a valid answer."), videos=[]);
        }
    }

    can reply_tools with ToolReply entry {
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
