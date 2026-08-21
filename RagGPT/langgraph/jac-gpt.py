"""Jac-GPT core, LangGraph port: RAG-grounded, LLM-routed multi-agent chat over the Jac docs.

A JacGPTFactory owns one shared RagEngine and one JacGPT instance per session id.
Each JacGPT compiles a StateGraph whose entry point is a conditional edge: an LLM
router picks exactly one specialist agent, that agent answers, and the graph ends.
The doc-grounded agents (RagChat, CodingChat, DebuggerChat) answer through ReAct
tool calls into the shared RagEngine; QAChat and OffTopicChat are plain LLM calls.

Agent prompts, descriptions and decoding settings live in prompts.py.

Run `python jac-gpt.py` for a terminal chat.
"""

from rag_engine import RagEngine
from prompts import AGENTS, ROUTER_INTENT, RouteDecision, SEARCH_DOCS_DESCRIPTION
from typing import Optional, List, Dict, Any, TypedDict
import os
import secrets
import time

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

# override=False: the shell wins, so an exported OPENAI_BASE_URL/MODEL points
# this arm at the same server as its siblings rather than being overridden by
# a committed .env.
load_dotenv(override=False)

# Jac passes chat_history[-10:] to each agent; one entry here holds a user/bot pair.
HISTORY_TURNS = 5


class ChatState(TypedDict):
    """State threaded through the graph for one user turn."""
    message: str
    chat_history: List[Dict[str, str]]
    agent: str
    response: str


class SessionInfo:
    def __init__(self):
        self.session_id: str = secrets.token_hex(16)  # Generate a random session ID
        self.chat_history: list[dict[str, str]] = []
        self.created_at: str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.updated_at: str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def update_history(self, user_message: str, bot_response: str):
        self.chat_history.append({"user": user_message, "bot": bot_response})
        self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _bare_model(name: str) -> str:
    """Drop any provider prefix: "openai/x", "openai:x" and "x" all mean x."""
    return name.replace(":", "/").split("/")[-1]


class JacGPT:
    def __init__(self, session_id: str, rag_engine: RagEngine):
        self.session_id = session_id
        self.session_info: SessionInfo = SessionInfo()
        self.rag_engine = rag_engine
        # $MODEL is shared with the byLLM siblings, which need litellm's
        # "openai/" prefix on an unfamiliar name. This side puts the string on
        # the wire as the model id, so the prefix is stripped, not passed on.
        self.model_name: str = _bare_model(os.environ.get("MODEL", "") or rag_engine.config.model_name)

        @tool(description=SEARCH_DOCS_DESCRIPTION)
        def search_docs(query: str) -> str:
            return self.rag_engine(query)

        self.search_docs = search_docs
        self.router_llm = ChatOpenAI(
            model=self.model_name, temperature=0
        ).with_structured_output(RouteDecision)
        self.llms: Dict[str, ChatOpenAI] = {
            name: ChatOpenAI(model=self.model_name, temperature=spec.temperature)
            for name, spec in AGENTS.items()
        }
        self.react_agents = {
            name: create_react_agent(self.llms[name], [self.search_docs], prompt=spec.prompt)
            for name, spec in AGENTS.items() if spec.use_tools
        }

        graph = StateGraph(ChatState)
        graph.add_node("RagChat", self.RagChat)
        graph.add_node("QAChat", self.QAChat)
        graph.add_node("CodingChat", self.CodingChat)
        graph.add_node("DebuggerChat", self.DebuggerChat)
        graph.add_node("OffTopicChat", self.OffTopicChat)
        # The router is the entry-point edge itself, mirroring `visit [-->] by llm(select=1)`.
        graph.add_conditional_edges(
            START, lambda state: self.route(state["message"]), {name: name for name in AGENTS}
        )
        for name in AGENTS:
            graph.add_edge(name, END)
        self.graph = graph.compile()

    def route(self, query: str) -> str:
        try:
            agent = self.router_llm.invoke([SystemMessage(ROUTER_INTENT), HumanMessage(query)]).agent #type: ignore
        except Exception as e:
            print(f"Routing failed ({e}); defaulting to RagChat")
            agent = "RagChat"
        print(f"Routed to {agent}")
        return agent

    def _respond(self, state: ChatState, name: str) -> Dict[str, str]:
        """Run one specialist agent over the turn; shared by the five node methods below."""
        spec = AGENTS[name]
        print(f"Entering {name} node")

        messages: List[Any] = []
        for turn in state["chat_history"]:
            messages.append(HumanMessage(turn["user"]))
            messages.append(AIMessage(turn["bot"]))
        messages.append(HumanMessage(state["message"]))

        if spec.use_tools:
            try:
                result = self.react_agents[name].invoke(
                    {"messages": messages}, config={"recursion_limit": spec.recursion_limit}
                )
                response = str(result["messages"][-1].content)
            except GraphRecursionError:
                response = f"Stopped after {spec.max_iterations} tool iterations without reaching a final answer."
        else:
            response = str(self.llms[name].invoke([SystemMessage(spec.prompt)] + messages).content)

        return {"agent": name, "response": response}

    def RagChat(self, state: ChatState) -> Dict[str, str]:
        return self._respond(state, "RagChat")

    def QAChat(self, state: ChatState) -> Dict[str, str]:
        return self._respond(state, "QAChat")

    def CodingChat(self, state: ChatState) -> Dict[str, str]:
        return self._respond(state, "CodingChat")

    def DebuggerChat(self, state: ChatState) -> Dict[str, str]:
        return self._respond(state, "DebuggerChat")

    def OffTopicChat(self, state: ChatState) -> Dict[str, str]:
        return self._respond(state, "OffTopicChat")

    def interact(self, message: str) -> Dict[str, str]:
        """Run one user turn through the graph and record it in the session history."""
        state = self.graph.invoke({
            "message": message,
            "chat_history": self.session_info.chat_history[-HISTORY_TURNS:],
            "agent": "",
            "response": "",
        })
        response = state.get("response") or "Sorry, I could not produce a response. Please try again."
        self.session_info.update_history(message, response)
        return {"session_id": self.session_id, "agent": state.get("agent", ""), "response": response}


class JacGPTFactory:
    def __init__(self, config_path: str = "config.json"):
        self.rag_engine = RagEngine(config_path=config_path)
        self.jac_gpt_instances: Dict[str, JacGPT] = {}  # session_id -> JacGPT instance

    def get_instance(self, session_id: str) -> JacGPT:
        """Return this session's JacGPT, creating one on the shared engine if it is new."""
        if session_id not in self.jac_gpt_instances:
            self.jac_gpt_instances[session_id] = JacGPT(session_id, self.rag_engine)
            print(f"Session created: {session_id}")
        return self.jac_gpt_instances[session_id]

    def interact(self, message: str, session_id: str = "default") -> Dict[str, str]:
        return self.get_instance(session_id).interact(message)

    def get_session(self, session_id: str = "default") -> Dict[str, Any]:
        instance = self.jac_gpt_instances.get(session_id)
        if instance is None:
            return {"session_id": session_id, "found": False, "chat_history": []}
        info = instance.session_info
        return {
            "session_id": instance.session_id,
            "found": True,
            "chat_history": info.chat_history,
            "created_at": info.created_at,
            "updated_at": info.updated_at,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Jac-GPT core — ask about the Jac language. Type /exit to quit.")
    print("=" * 60)
    factory = JacGPTFactory()
    cli_session = f"cli_{int(time.time())}"
    while True:
        try:
            user_input = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            break
        msg = user_input.strip()
        if not msg:
            continue
        if msg in ["/exit", "/quit", "exit", "quit"]:
            break
        try:
            payload = factory.interact(message=msg, session_id=cli_session)
            print(f"\n[{payload.get('agent', 'unknown')}]")
            print(payload.get("response", "(no response)"))
        except Exception as e:
            print(f"error: {e}")
    print("bye.")
