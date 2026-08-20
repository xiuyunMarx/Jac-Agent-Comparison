"""Jac-GPT core: RAG-grounded, LLM-routed multi-agent chat over the Jac docs.

An `interact` walker routes each message through a Router node to the
specialist agent node that fits it best (docs Q&A, code writing, debugging,
small talk, off-topic). Doc-grounded agents answer via ReAct tool calls into
the FAISS-backed RAG engine. Sessions persist as graph nodes.

Agent prompts, the tool description and the router intent live in main.impl.jac,
which the compiler pairs with this file by basename.

Run `jac run main.jac` for a terminal chat, or `jac start main.jac` to expose
the walkers as REST endpoints (POST /walker/interact, /walker/get_session).
"""

import os;
import time;
import from datetime { datetime }
import from jaclang.byllm.lib { Model }
import from dotenv { load_dotenv }
import from rag_engine { RagEngine, AppConfig, load_config }

# Load .env (API keys, MODEL override) before any LLM-backed initialization.
with entry {
    load_dotenv(override=True);
}

glob app_config: AppConfig = load_config("faiss_reranking");
# vLLM backend. The endpoint travels via litellm's HOSTED_VLLM_API_BASE env var:
# byllm's litellm path drops config base_url (model.impl.jac pops api_base from
# params without forwarding it), so the env var is the only working channel.
with entry {
    os.environ.setdefault("HOSTED_VLLM_API_BASE", "http://127.0.0.1:8001/v1");
    os.environ.setdefault("HOSTED_VLLM_API_KEY", "EMPTY");
}
# ctx_window: litellm has no metadata for hosted_vllm/* models; without this
# byllm's auto-compaction misreads the window and errors on tiny histories.
# Must match serve_vllm.sh --max-model-len.
glob llm = Model(
    model_name=os.environ.get("MODEL", app_config.model_name),
    ctx_window=16384
);

glob base_dir: str = os.path.dirname(__file__);
glob rag_engine: RagEngine = RagEngine(
    file_path=os.path.join(base_dir, app_config.docs_path),
    faiss_path=os.path.join(base_dir, app_config.faiss_path),
    chunk_size=app_config.chunk_size,
    chunk_overlap=app_config.chunk_overlap,
    chunk_nos=app_config.chunk_k,
    top_n=app_config.rerank_top_n,
    cross_encoder_model=app_config.cross_encoder_model,
    embedding_model=app_config.embedding_model
);


def search_docs(query: str) -> str {
    return rag_engine.search(query=query);
}


node Router {}


node RagChat {
    def respond(message: str, chat_history: list[dict[str, str]]) -> str by llm(
        tools=[search_docs],
        temperature=0.5,
        max_react_iterations=6
    );

    can chat with interact entry {
        print("Entering RAG node");
        visitor.agent = "RagChat";
        visitor.response = self.respond(
            message=visitor.message,
            chat_history=visitor.chat_history[-10:]
        );
    }
}


node CodingChat {
    def code_jac(message: str, chat_history: list[dict[str, str]]) -> str by llm(
        tools=[search_docs],
        temperature=0.3,
        max_react_iterations=20
    );

    can chat with interact entry {
        print("Entering Coding node");
        visitor.agent = "CodingChat";
        visitor.response = self.code_jac(
            message=visitor.message,
            chat_history=visitor.chat_history[-10:]
        );
    }
}


node DebuggerChat {
    def debug_code(message: str, chat_history: list[dict[str, str]]) -> str by llm(
        tools=[search_docs],
        max_react_iterations=12
    );

    can chat with interact entry {
        print("Entering Debugger node");
        visitor.agent = "DebuggerChat";
        visitor.response = self.debug_code(
            message=visitor.message,
            chat_history=visitor.chat_history[-10:]
        );
    }
}


node QAChat {
    def respond(message: str, chat_history: list[dict[str, str]]) -> str by llm(
        temperature=0.3
    );

    can chat with interact entry {
        print("Entering QA node");
        visitor.agent = "QAChat";
        visitor.response = self.respond(
            message=visitor.message,
            chat_history=visitor.chat_history[-10:]
        );
    }
}


node OffTopicChat {
    def respond(message: str, chat_history: list[dict[str, str]]) -> str by llm(
        temperature=0.3
    );

    can chat with interact entry {
        print("Entering Off-Topic node");
        visitor.agent = "OffTopicChat";
        visitor.response = self.respond(
            message=visitor.message,
            chat_history=visitor.chat_history[-10:]
        );
    }
}


node Session {
    has id: str;
    has chat_history: list[dict[str, str]] = [];
    has created_at: str = "";
    has updated_at: str = "";
}


walker :pub interact {
    has message: str;
    has session_id: str = "default";
    has chat_history: list[dict[str, str]] = [];
    has response: str = "";
    has agent: str = "";

    can setup with Root entry {
        # Find or create the session, record the user turn
        now = datetime.now().isoformat();
        matches = [s for s in [-->][?:Session] if s.id == self.session_id];
        if matches {
            target = matches[0];
        } else {
            target = Session(id=self.session_id, created_at=now);
            here ++> target;
            print("Session created:", self.session_id);
        }
        target.chat_history.append({"role": "user", "content": self.message});
        target.updated_at = now;
        self.chat_history = target.chat_history;

        # Find or create the router with its agent nodes
        routers = [-->][?:Router];
        if routers {
            visit routers[0];
        } else {
            router = Router();
            here ++> router;
            router ++> RagChat();
            router ++> CodingChat();
            router ++> DebuggerChat();
            router ++> QAChat();
            router ++> OffTopicChat();
            visit router;
        }
    }

    can route with Router entry {
        visit [-->] by llm(
            select=1,
            intent=ROUTER_INTENT
        );
    }

    can finalize with Root exit {
        if not self.response {
            self.response = "Sorry, I could not produce a response. Please try again.";
        }
        for s in [-->][?:Session] {
            if s.id == self.session_id {
                s.chat_history.append({"role": "assistant", "content": self.response});
                s.updated_at = datetime.now().isoformat();
                break;
            }
        }
        report {
            "session_id": self.session_id,
            "agent": self.agent,
            "response": self.response
        };
    }
}


walker :pub get_session {
    has session_id: str = "default";

    can fetch with Root entry {
        for s in [-->][?:Session] {
            if s.id == self.session_id {
                report {
                    "session_id": s.id,
                    "found": True,
                    "chat_history": s.chat_history,
                    "created_at": s.created_at,
                    "updated_at": s.updated_at
                };
                return;
            }
        }
        report {"session_id": self.session_id, "found": False, "chat_history": []};
    }
}


with entry:__main__ {
    print("=" * 60);
    print("Jac-GPT core — ask about the Jac language. Type /exit to quit.");
    print("=" * 60);
    cli_session = f"cli_{int(time.time())}";
    running = True;
    while running {
        try {
            user_input = input("\nyou> ");
        } except EOFError {
            break;
        } except KeyboardInterrupt {
            break;
        }
        msg = user_input.strip();
        if not msg {
            continue;
        }
        if msg in ["/exit", "/quit", "exit", "quit"] {
            break;
        }
        try {
            result = root spawn interact(message=msg, session_id=cli_session);
            payload = result.reports[0] if result.reports else {};
            print(f"\n[{payload.get('agent', 'unknown')}]");
            print(payload.get("response", "(no response)"));
        } except Exception as e {
            print(f"error: {e}");
        }
    }
    print("bye.");
}
