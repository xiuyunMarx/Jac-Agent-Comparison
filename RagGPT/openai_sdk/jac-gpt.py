"""Jac-GPT core, no-framework port: RAG-grounded, LLM-routed multi-agent chat
over the Jac docs — no langchain, no langgraph, no byllm; just the `openai`
package and a `while` loop.

A JacGPTFactory owns one shared RagEngine, one OpenAI client, and one JacGPT
per session id. Each user turn is routed by a plain chat.completions call
constrained to the five agent names (what `with_structured_output` /
`visit [-->] by llm(select=1)` were doing), then answered by the selected
specialist: the doc-grounded agents (RagChat, CodingChat, DebuggerChat) run a
hand-rolled ReAct loop over the single search_docs tool, QAChat and
OffTopicChat are one completion each.

Agent prompts, descriptions and decoding settings live in prompts.py and are
verbatim identical to the other systems' — the eval's controlled variable.

The file name and the exported JacGPTFactory(...).interact(message=,
session_id=) contract match the LangGraph side exactly, so the eval harness
drives both through the same in-process driver. The two sibling modules are
loaded by absolute path under unique names because that driver may exec this
file into a process that already imported the LangGraph side's `prompts` and
`rag_engine`.

Run `python jac-gpt.py` for a terminal chat.
"""

import importlib.util
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

_HERE = Path(__file__).resolve().parent


def _load_sibling(stem: str):
    """Import ./{stem}.py under the collision-proof name openai_sdk_{stem}."""
    name = f"openai_sdk_{stem}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _HERE / f"{stem}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


prompts = _load_sibling("prompts")
rag = _load_sibling("rag_engine")

AGENTS = prompts.AGENTS
ROUTER_INTENT = prompts.ROUTER_INTENT
ROUTER_RESPONSE_FORMAT = prompts.ROUTER_RESPONSE_FORMAT
SEARCH_DOCS_TOOL = prompts.SEARCH_DOCS_TOOL

# Load ../.env (API key, MODEL override) before any LLM-backed initialization,
# exactly as the siblings do. OPENAI_API_KEY and OPENAI_BASE_URL are then read
# natively by the openai SDK — the token-counting eval proxy needs nothing more.
# override=False: the shell wins, so an exported OPENAI_BASE_URL/MODEL points
# this arm at the same server as its siblings.
load_dotenv(_HERE.parent / ".env", override=False)

# Jac passes chat_history[-10:] to each agent; one entry here holds a user/bot pair.
HISTORY_TURNS = 5

DEFAULT_CONFIG_PATH = str(_HERE / "config" / "faiss_reranking.json")


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
    def __init__(self, session_id: str, rag_engine, client: OpenAI):
        self.session_id = session_id
        self.session_info = SessionInfo()
        self.rag_engine = rag_engine
        self.client = client
        # $MODEL is shared with the byLLM siblings, which need litellm's
        # "openai/" prefix on an unfamiliar name. This side puts the string on
        # the wire as the model id, so the prefix is stripped, not passed on.
        self.model_name: str = _bare_model(os.environ.get("MODEL", "") or rag_engine.config.model_name)

    # -- the router: one completion constrained to the five agent names --------

    def route(self, query: str) -> str:
        """Pick the specialist for this message; RagChat on any failure, as the
        LangGraph side does. The router deliberately sees no chat history
        (matching that side; a known asymmetry the eval measures)."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                messages=[{"role": "system", "content": ROUTER_INTENT},
                          {"role": "user", "content": query}],
                response_format=ROUTER_RESPONSE_FORMAT,
            )
            agent = json.loads(response.choices[0].message.content)["agent"]
            if agent not in AGENTS:
                raise ValueError(f"router picked unknown agent {agent!r}")
        except Exception as e:
            print(f"Routing failed ({e}); defaulting to RagChat")
            agent = "RagChat"
        print(f"Routed to {agent}")
        return agent

    # -- the specialists -------------------------------------------------------

    def respond(self, name: str, message: str, chat_history: List[Dict[str, str]]) -> str:
        """Run one specialist agent over the turn: system prompt, the last
        HISTORY_TURNS user/bot pairs, the current message."""
        spec = AGENTS[name]
        print(f"Entering {name} node")

        messages: List[Dict[str, Any]] = [{"role": "system", "content": spec.prompt}]
        for turn in chat_history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["bot"]})
        messages.append({"role": "user", "content": message})

        if spec.use_tools:
            return self.react(spec, messages)
        response = self.client.chat.completions.create(
            model=self.model_name, temperature=spec.temperature, messages=messages)
        return str(response.choices[0].message.content or "")

    def react(self, spec, messages: List[Dict[str, Any]]) -> str:
        """The ReAct loop create_react_agent was providing: call the model with
        the tool schema; execute and append every requested call; repeat until
        it answers in prose.

        Budget semantics match the LangGraph side's recursion_limit of
        2*max_iterations + 1 supersteps: up to max_iterations tool batches plus
        one final model call, and a run still asking for tools past that ends
        with the same canned message its GraphRecursionError handler produced.
        """
        for iteration in range(spec.max_iterations + 1):
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=spec.temperature,
                messages=messages,
                tools=[SEARCH_DOCS_TOOL],
            )
            reply = response.choices[0].message
            tool_calls = reply.tool_calls or []
            if not tool_calls:
                return str(reply.content or "")
            if iteration == spec.max_iterations:
                break
            messages.append({
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [
                    {"id": call.id, "type": "function",
                     "function": {"name": call.function.name,
                                  "arguments": call.function.arguments}}
                    for call in tool_calls
                ],
            })
            for call in tool_calls:  # a `for` loop is already serial
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": self.run_tool(call)})
        return f"Stopped after {spec.max_iterations} tool iterations without reaching a final answer."

    def run_tool(self, call) -> str:
        """Execute one requested tool call; errors come back as tool output for
        the model to react to, as langgraph's ToolNode arranged."""
        if call.function.name != "search_docs":
            return f"Error: unknown tool {call.function.name!r}."
        try:
            args = json.loads(call.function.arguments or "{}")
            return self.rag_engine.search(query=args["query"])
        except Exception as e:
            return f"Error: {e!r}. Please fix your mistakes."

    # -- one user turn ---------------------------------------------------------

    def interact(self, message: str) -> Dict[str, str]:
        """Route the message, run the chosen agent, record the turn."""
        chat_history = self.session_info.chat_history[-HISTORY_TURNS:]
        agent = self.route(message)
        response = self.respond(agent, message, chat_history)
        if not response:
            response = "Sorry, I could not produce a response. Please try again."
        self.session_info.update_history(message, response)
        return {"session_id": self.session_id, "agent": agent, "response": response}


class JacGPTFactory:
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.rag_engine = rag.RagEngine(config_path=config_path)
        # One client for every session: OPENAI_API_KEY / OPENAI_BASE_URL are
        # read from the environment natively, and deliberately not named here.
        self.client = OpenAI()
        self.jac_gpt_instances: Dict[str, JacGPT] = {}  # session_id -> JacGPT instance

    def get_instance(self, session_id: str) -> JacGPT:
        """Return this session's JacGPT, creating one on the shared engine if it is new."""
        if session_id not in self.jac_gpt_instances:
            self.jac_gpt_instances[session_id] = JacGPT(session_id, self.rag_engine, self.client)
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
