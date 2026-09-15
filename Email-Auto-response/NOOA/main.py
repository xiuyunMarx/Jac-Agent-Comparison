"""Email auto-response agent in NVIDIA Object-Oriented Agents (NOOA).

Same task, mailbox, knobs and output as ../byLLM/nodes.jac, written the way
NOOA's docs say to write an agent: one Python class, no graph, no walker.

    EmailAgent(Agent)
        web_search(query) -> str                  regular method: the one LLM-facing tool
        filter_emails(abstract) -> Triage           `...`  PredictStrategy   (byLLM: by llm())
        email_action_agent(thread) -> ThreadAnalysis  `...`  CodeActStrategy (byLLM: by llm(tools=[web_search]))
        email_response_writer(analysis, thread) -> DraftReply  `...`  CodeActStrategy (same)
        run() -> list[dict]                       hidden, deterministic: the walker's loop

The `sem` strings ride as docstrings, `Annotated[T, "..."]` parameter
descriptions and Pydantic `Field(description=...)`; the return annotations are
the output contracts NOOA validates. A stage with tools is a CodeAct method:
the model works in a Python REPL where `self.web_search(...)` is the visible
capability (NOOA's tool layer is the object's public methods), and finishes
with `return_result(...)`, which NOOA validates against the return type --
byllm's finish-tool ReAct loop, in NOOA's shape. `max_iterations=10` is
byLLM's `max_react_iterations=10`.

Model: $BENCH_MODEL (bare id, default glm-5.2) as litellm `openai/<id>` on
$OPENAI_BASE_URL (default https://ollama.com/v1) with $OPENAI_API_KEY --
what nodes.jac does. Temperature 0.7 and no max_tokens: byllm's default call
params, matched, as the openai_sdk arm matches them. Token accounting is the
harness's: MockMailbox installs the shared meter on the openai SDK, which
litellm (and therefore NOOA) calls, so usage lands in results_<case_id>.json
with no agent-side counting, exactly as for the other arms.

Usage: python main.py [path/to/dataset.json]     ($EMAIL_DATASET honoured too)
Results: mock_output/results_<case_id>.json      ($EMAIL_OUTPUT_DIR overrides the dir)
BENCH_TRACE=<file.jsonl> adds one row per model call (messages, reply, usage).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field

from nooa import Agent, EventQuery, hidden, strategy

# Plumbing the model must not see: NOOA exposes every module-level name to
# CodeAct-generated code unless it is hidden (AGENTS.md, "Visibility"). The
# types above stay visible because the agent's signatures reference them.
with hidden:
    import asyncio
    import json
    import os
    import sys
    import time
    from contextvars import ContextVar
    from pathlib import Path

    from nooa.config import CodeActConfig, PredictConfig
    from nooa.config.truncation_config import FormatConfig, TruncationConfig
    from nooa.strategies import CodeActStrategy, PredictStrategy
    from nooa.unifiedllm import CompletionClient, LLMResponse

    ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT.parent))  # make the shared mock_mailbox importable


@hidden
def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines into the environment, never overriding it (openai_sdk's loader)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value:
            os.environ.setdefault(key, value)


load_dotenv(ROOT / ".env")

with hidden:
    from mock_mailbox import MockMailbox  # noqa: E402  (needs the sys.path line)
# Note: the class name still shows in CodeAct's execution context, because NOOA
# auto-imports the classes of the agent's attribute values (self._mailbox) so
# generated code can doc()/isinstance() them. The instance itself is private.

# ---------------------------------------------------------------------------
# Configuration: nodes.jac's `glob` block.
# ---------------------------------------------------------------------------

BENCH_MODEL: Annotated[str, hidden] = os.environ.get("BENCH_MODEL", "glm-5.2")
MODEL_NAME: Annotated[str, hidden] = BENCH_MODEL if "/" in BENCH_MODEL else f"openai/{BENCH_MODEL}"
MODEL_BASE: Annotated[str, hidden] = os.environ.get("OPENAI_BASE_URL", "https://ollama.com/v1")
API_KEY: Annotated[str, hidden] = os.environ.get("OPENAI_API_KEY", "")
# byllm's default call params (jac.toml declares none): temperature 0.7, no max_tokens.
TEMPERATURE: Annotated[float, hidden] = 0.7
# gpt-5 / o-series bill their thinking against the completion budget; "minimal"
# switches that off. drop_params lets litellm drop the temperature they reject.
REASONING_MODEL: Annotated[bool, hidden] = BENCH_MODEL.split("/", 1)[-1].startswith(("gpt-5", "o1", "o3", "o4"))
LLM_CALL_PARAMS: Annotated[dict[str, Any], hidden] = {"reasoning_effort": "minimal"} if REASONING_MODEL else {}
# byLLM: max_react_iterations=10 on the two tool-using stages; byllm's typed
# output loop allows max_output_retries=3, i.e. four attempts.
MAX_REACT_ITERATIONS: Annotated[int, hidden] = 10
OUTPUT_ATTEMPTS: Annotated[int, hidden] = 4
TRACE_PATH: Annotated[str, hidden] = os.environ.get("BENCH_TRACE", "")


# ---------------------------------------------------------------------------
# Typed values: tools.jac's Email/Thread and nodes.jac's obj/enum declarations,
# field for field; the `sem` strings are the Field descriptions.
# ---------------------------------------------------------------------------


class Email(BaseModel):
    subject: str
    body: str
    sender: str
    thread_id: str


class Thread(BaseModel):
    thread_id: str
    subject: str
    sender: str
    recipient: str
    emails: list[Email]


class MailAbstract(BaseModel):
    id: str
    thread_id: str
    snippet: str
    sender: str


class Classification(str, Enum):
    IMPORTANT = "IMPORTANT"
    SPAM = "SPAM"
    NEWSLETTER = "NEWSLETTER"
    PROMOTIONAL = "PROMOTIONAL"
    NOTIFICATIONS = "NOTIFICATIONS"
    SOCIAL = "SOCIAL"
    UPDATES = "UPDATES"


class Triage(BaseModel):
    """The filter's verdict for one abstract. A named model rather than a bare enum: NOOA wraps a
    bare enum in a one-field model whose key only appears in the response_format schema, which
    ollama.com ignores for GLM; the class rendered in the prompt is what the model reads."""

    classification: Classification = Field(description="The one category this email belongs to.")


class ThreadAnalysis(BaseModel):
    thread_id: str
    summary: str = Field(description="Concise summary of the email thread's context and overall sentiment.")
    main_points: list[str] = Field(description="The main queries or concerns a reply must address.")
    sender_email: str = Field(description="Bare email address of the person to reply to, e.g. name@example.com.")
    communication_style: str = Field(description="Tone and style used in the thread, e.g. formal, friendly, terse.")


class DraftReply(BaseModel):
    recipient: str = Field(description="Bare email address of the person being replied to.")
    subject: str = Field(description="Subject line for the reply, matching the thread's subject.")
    message: str = Field(description="Full body of the reply email, written in the user's voice.")


# ---------------------------------------------------------------------------
# The model seam: NOOA's litellm CompletionClient. Token/cost accounting is the
# harness's openai-SDK meter, untouched; the subclass only writes the optional
# BENCH_TRACE rows (stage = the agentic method being generated).
# ---------------------------------------------------------------------------

_stage: Annotated[ContextVar[str], hidden] = ContextVar("email_stage", default="?")


@hidden
def _content_len(content: Any) -> int:
    return len(content) if isinstance(content, str) else len(json.dumps(content, ensure_ascii=False))


@hidden
class TracedClient(CompletionClient):
    async def acall(self, messages: list[dict[str, Any]], *args: Any, **kwargs: Any) -> LLMResponse:
        response = await super().acall(messages, *args, **kwargs)
        if TRACE_PATH:
            reply = response.assistant_message.get("content")
            reply = reply if isinstance(reply, str) else ("" if reply is None else json.dumps(reply))
            row = {
                "ts": time.time(),
                "stage": _stage.get(),
                "model": BENCH_MODEL,
                "prompt_chars": sum(_content_len(m.get("content") or "") for m in messages),
                "reply_chars": len(reply),
                "tool_calls": [tc.name for tc in response.tool_calls],
                "usage": response.usage,
                "messages": messages,
                "reply": reply,
            }
            with open(TRACE_PATH, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return response


with hidden:
    llm = TracedClient(
        model=MODEL_NAME,
        api_base=MODEL_BASE,
        api_key=API_KEY or "unset",
        temperature=TEMPERATURE,
        drop_params=True,
        **LLM_CALL_PARAMS,
    )


@hidden
def by_llm() -> PredictStrategy:
    """`by llm()`: one structured call, byllm's output retries."""
    return PredictStrategy(config=PredictConfig(temperature=TEMPERATURE, max_retries=OUTPUT_ATTEMPTS))


@hidden
def by_llm_with_tools() -> CodeActStrategy:
    """`by llm(tools=[web_search], max_react_iterations=10)`: a tool loop that ends in a typed result."""
    return CodeActStrategy(
        config=CodeActConfig(temperature=TEMPERATURE, max_iterations=MAX_REACT_ITERATIONS)
    )


# Arguments (threads, analyses) are rendered whole, never cut.
UNTRUNCATED: Annotated[TruncationConfig, hidden] = TruncationConfig(prefill_format=FormatConfig(max_string=None, max_length=None, max_depth=None))


# ---------------------------------------------------------------------------
# The agent.
# ---------------------------------------------------------------------------


class EmailAgent(Agent, llm=llm, event_query=EventQuery.current_call(), truncation=UNTRUNCATED):
    """An email assistant working the mailbox owner's inbox: it screens each new thread,
    analyzes the ones that need a personal reply, and drafts that reply in the owner's voice."""

    # The walker's fields: bookkeeping for main(), hidden from the model.
    abstracts: Annotated[list[MailAbstract], hidden]
    drafted: Annotated[list[dict], hidden]

    def __init__(self, mailbox: Any, **kwargs: Any):   # a MockMailbox; typed Any so the class stays hidden
        super().__init__(**kwargs)
        self._mailbox = mailbox          # private: the model sees web_search(), not the mailbox
        self.abstracts = []
        self.drafted = []

    # -- the one LLM-facing tool (byLLM: tools=[web_search]) -----------------

    def web_search(self, query: Annotated[str, "The search query."]) -> str:
        """Search the internet for information about a topic and return relevant results."""
        return str(self._mailbox.web_search(query))

    # -- node check_new_emails: mechanical, no LLM ---------------------------

    @hidden
    def fetch_mail_abstracts(self) -> list[MailAbstract]:
        """One abstract per unique thread, self-sent mail excluded."""
        owner = str(self._mailbox.owner_email or "")
        abstracts: list[MailAbstract] = []
        seen_threads: list[str] = []
        for content in self._mailbox.search():
            tid = str(content["threadId"])
            sender = str(content["sender"])
            if tid in seen_threads:
                continue
            if owner != "" and owner in sender:
                continue
            seen_threads.append(tid)
            abstracts.append(
                MailAbstract(id=str(content["id"]), thread_id=tid, snippet=str(content["snippet"]), sender=sender)
            )
        return abstracts

    @hidden
    def get_thread(self, thread_id: str) -> Thread:
        """tools.jac's MailBox.get_thread: the typed thread; the request is recorded for eval."""
        self._mailbox.get_thread(thread_id)
        tid = str(thread_id).strip().strip("'\"")
        for e in self._mailbox.emails:
            if e["threadId"] == tid:
                return Thread(
                    thread_id=tid,
                    subject=str(e.get("subject", "")),
                    sender=str(e["sender"]),
                    recipient=str(self._mailbox.owner_email),
                    emails=[
                        Email(subject=str(e.get("subject", "")), body=str(m["body"]), sender=str(m["from"]), thread_id=tid)
                        for m in e["full_thread"]
                    ],
                )
        return Thread(thread_id=tid, subject="", sender="", recipient="", emails=[])

    # -- node draft_responses: the three judgments ---------------------------

    @hidden
    @strategy(by_llm())
    async def filter_emails(self, abstract: MailAbstract) -> Triage:
        """Senior Email Analyst: classify the email from its snippet and sender. Only messages
        actually directed at the user that need a personal reply are IMPORTANT; newsletters are
        NEWSLETTER, promotional content is PROMOTIONAL, automated notifications are NOTIFICATIONS,
        social platform mail is SOCIAL, product/service updates are UPDATES, junk is SPAM."""
        ...

    @hidden
    @strategy(by_llm_with_tools())
    async def email_action_agent(self, thread: Thread) -> ThreadAnalysis:
        """Email Action Specialist: analyze the complete email thread to understand its context,
        key points and sentiment. Identify the main query the reply must address, the sender's
        email address and the communication style of the thread. You may call web_search when the
        reply will need facts the thread does not contain. Search only for what the analysis needs,
        never repeat a search on the same topic, and do not narrate."""
        ...

    @hidden
    @strategy(by_llm_with_tools())
    async def email_response_writer(self, analysis: ThreadAnalysis, thread: Thread) -> DraftReply:
        """Email Response Writer: draft a reply to the thread, assuming the persona of the user
        (the thread's recipient) and mimicking the communication style of the thread. Address
        every main point from the analysis. Research the topic with web_search first IF
        NECESSARY -- if research is needed, do it BEFORE drafting the response, at most once per
        topic; never repeat a search whose result you already have."""
        ...

    # -- walker EmailAgent: the workflow, in Python --------------------------

    @hidden
    async def run(self) -> list[dict]:
        """check -> per thread: classify, (skip unless IMPORTANT), analyze, write, file the draft.
        The walker's skip / record / one-retry decisions, verbatim."""
        print("# Checking for new emails")
        self.abstracts = self.fetch_mail_abstracts()
        if len(self.abstracts) == 0:
            print("## No new emails")
            return self.drafted
        print(f"## {len(self.abstracts)} new email threads")

        for abstract in self.abstracts:
            _stage.set("filter_emails")
            category = (await self.filter_emails(abstract)).classification
            print(f"### {abstract.thread_id} [{abstract.sender}] -> {category.value}")
            if category != Classification.IMPORTANT:
                continue
            thread = self.get_thread(abstract.thread_id)
            if len(thread.emails) == 0:
                continue

            # A stage that fails its contract (NOOA raises once its retries are
            # spent) is a failure of this one email, recorded like byLLM's
            # unstructured answer; it must not forfeit the rest of the inbox.
            _stage.set("email_action_agent")
            try:
                analysis = await self.email_action_agent(thread)
            except Exception as exc:  # noqa: BLE001 - recorded, then the next email
                error = f"email_action_agent raised {type(exc).__name__}: {str(exc)[:500]}"
                self._mailbox.record_draft_error("", error)
                print(f"### {abstract.thread_id}: {error}")
                continue

            # One retry, so a single slip does not cost a whole email.
            _stage.set("email_response_writer")
            reply: DraftReply | None = None
            failure = ""
            for attempt in range(2):
                try:
                    reply = await self.email_response_writer(analysis, thread)
                    break
                except Exception as exc:  # noqa: BLE001
                    reply = None
                    failure = f"email_response_writer raised {type(exc).__name__}: {str(exc)[:500]}"
                if attempt == 0:
                    print(f"### {abstract.thread_id}: writer failed ({failure[:80]}), retrying")
            if reply is None:
                self._mailbox.record_draft_error("", failure)
                print(f"### {abstract.thread_id}: {failure}")
                continue

            self._mailbox.create_draft(reply.recipient, reply.subject, reply.message)
            self.drafted.append({"to": reply.recipient, "subject": reply.subject})
            print(f"### Draft created for {reply.recipient}: {reply.subject}")
        return self.drafted


@hidden
def main() -> int:
    dataset = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("EMAIL_DATASET", "")
        or str(ROOT.parent / "mock_mailbox" / "datasets" / "batch_001.json")
    )
    # The mailbox installs the shared token meter on the openai SDK; NOOA's
    # litellm client goes through that SDK, so every call is counted.
    mailbox = MockMailbox(dataset)
    print(f"# Loaded mock mailbox '{mailbox.case_id}' ({len(mailbox.emails)} emails) from {dataset}")

    agent = EmailAgent(mailbox)
    # Never lose a batch's captured drafts (or its token stats) to a crash
    # partway through the inbox -- results are written either way.
    failure = ""
    try:
        asyncio.run(agent.run())
    except Exception as exc:  # noqa: BLE001 - reported, saved, then re-raised
        failure = f"{type(exc).__name__}: {exc}"
        print(f"!! Run aborted: {failure}")

    out_dir = Path(os.environ.get("EMAIL_OUTPUT_DIR", "") or (ROOT / "mock_output"))
    out_path = mailbox.save_results(out_dir / f"results_{mailbox.case_id}.json")
    print("=" * 60)
    print(f"Run complete for '{mailbox.case_id}'")
    print(f"Email threads passed to agents: {len(agent.abstracts)}")
    print(f"Drafts captured:                {len(agent.drafted)}")
    for d in agent.drafted:
        print(f"  - to: {d['to']} | subject: {d['subject']}")
    if mailbox.draft_errors:
        print(f"Draft tool errors:              {len(mailbox.draft_errors)}")
        for e in mailbox.draft_errors:
            print(f"  - {e['error']}")
    print(f"LLM usage: {mailbox.usage_line()}")
    print(f"Results saved to: {out_path}")
    if failure:
        raise RuntimeError(failure)  # non-zero exit so a sweep cannot hide it
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
