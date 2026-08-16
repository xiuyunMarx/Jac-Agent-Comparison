"""byLLM email auto-response agent over the shared mock mailbox (../mock_mailbox).

Mirrors the CrewAI-LangGraph reference: `check_new_emails` scans the inbox and
excludes self-sent mail (mechanical, no LLM); `draft_responses` runs the three
agent stages (classify -> analyze -> write) as `by llm` functions. Drafts are
captured by the harness and written to mock_output/results_<case_id>.json.
"""
import os;
import importlib;
import from tools {MailBox, Thread, REPO_ROOT}


# byllm lives at jaclang.byllm on mainline jac (bundled ELF, jaseci-gen-sem dev
# mode) but is the separate pip byllm package on released jaclang 0.16.x (conda
# jaseci env); resolve whichever this toolchain provides.
def _resolve_model() -> any {
    try {
        return importlib.import_module("jaclang.byllm.lib").Model;
    } except ImportError {
        return importlib.import_module("byllm.lib").Model;
    }
}

glob Model: any = _resolve_model();


# One model knob for both implementations: crewai resolves OPENAI_MODEL_NAME
# through litellm, so setting it once keeps the comparison apples-to-apples.
# Defaults to the gpt-4o this was built on.
def _model_name() -> str {
    name = str(os.environ.get("OPENAI_MODEL_NAME", "") or "gpt-4o");
    if "/" in name {
        return name;
    }
    return f"openai/{name}";
}

glob llm = Model(_model_name());
glob MAILBOX: MailBox | None = None;


# Canned web-search tool for the LLM stages; bound to the active mailbox
# by EmailAgent.postinit.
def web_search(query: str) -> str {
    if MAILBOX is None {
        return "No mailbox loaded.";
    }
    return MAILBOX.web_search(query);
}


# byllm's ReAct loop falls back to the model's raw text when the model answers
# in prose instead of calling the finish tool (byllm/llm.impl/basellm.impl.jac:
# 998, 1001), so a `by llm(tools=[...])` stage can hand back a str rather than
# the declared type. Stages are checked instead of trusted, and a stage that
# answers in prose is recorded as a draft error rather than crashing the batch.
def unstructured_error(value: any, stage: str, expected: str) -> str {
    return f"{stage} returned {type(value).__name__} instead of {expected}: {str(value)[:500]}";
}


obj MailAbstract {
    has id: str;
    has thread_id: str;
    has snippet: str;
    has sender: str;
}

obj ThreadAnalysis {
    has thread_id: str;
    has summary: str;
    has main_points: list[str];
    has sender_email: str;
    has communication_style: str;
}

obj DraftReply {
    has recipient: str;
    has subject: str;
    has message: str;
}

sem ThreadAnalysis.summary = "Concise summary of the email thread's context and overall sentiment.";
sem ThreadAnalysis.main_points = "The main queries or concerns a reply must address.";
sem ThreadAnalysis.sender_email = "Bare email address of the person to reply to, e.g. name@example.com.";
sem ThreadAnalysis.communication_style = "Tone and style used in the thread, e.g. formal, friendly, terse.";
sem DraftReply.recipient = "Bare email address of the person being replied to.";
sem DraftReply.subject = "Subject line for the reply, matching the thread's subject.";
sem DraftReply.message = "Full body of the reply email, written in the user's voice.";


node check_new_emails {
    # One abstract per unique thread, self-sent mail excluded (mirrors the
    # CrewAI check_email node).
    def fetch_mail_abstracts(mailbox: MailBox) -> list[MailAbstract] {
        owner = mailbox.owner_email();
        emails = mailbox.search();
        mail_abstracts: list[MailAbstract] = [];
        seen_threads: list[str] = [];
        for content in emails {
            tid = str(content["threadId"]);
            sender = str(content["sender"]);
            if tid in seen_threads {
                continue;
            }
            if owner != "" and owner in sender {
                continue;
            }
            seen_threads.append(tid);
            mail_abstracts.append(
                MailAbstract(
                    id=str(content["id"]),
                    thread_id=tid,
                    snippet=str(content["snippet"]),
                    sender=sender
                )
            );
        }
        return mail_abstracts;
    }
}


enum Classification {
    IMPORTANT = "IMPORTANT",
    SPAM = "SPAM",
    NEWSLETTER = "NEWSLETTER",
    PROMOTIONAL = "PROMOTIONAL",
    NOTIFICATIONS = "NOTIFICATIONS",
    SOCIAL = "SOCIAL",
    UPDATES = "UPDATES"
}


node draft_responses {
    def filter_emails(abstract: MailAbstract) -> Classification by llm();
    def email_action_agent(thread: Thread) -> ThreadAnalysis by llm(tools=[web_search]);
    def email_response_writer(analysis: ThreadAnalysis, thread: Thread) -> DraftReply by llm(tools=[web_search]);
}

sem draft_responses.filter_emails = "Senior Email Analyst: classify the email from its snippet and sender. Only messages actually directed at the user that need a personal reply are IMPORTANT; newsletters are NEWSLETTER, promotional content is PROMOTIONAL, automated notifications are NOTIFICATIONS, social platform mail is SOCIAL, product/service updates are UPDATES, junk is SPAM.";
sem draft_responses.email_action_agent = "Email Action Specialist: analyze the complete email thread to understand its context, key points and sentiment. Identify the main query the reply must address, the sender's email address and the communication style of the thread.";
sem draft_responses.email_response_writer = "Email Response Writer: draft a reply to the thread, assuming the persona of the user (the thread's recipient) and mimicking the communication style of the thread. Address every main point from the analysis. Research the topic with web_search first IF NECESSARY.";


walker EmailAgent {
    has dataset_path: str;
    has mailbox: MailBox by postinit;
    has abstracts: list[MailAbstract] = [];
    has drafted: list[dict] = [];

    def postinit {
        self.mailbox = MailBox(dataset_path=self.dataset_path);
        global MAILBOX;
        MAILBOX = self.mailbox;
    }

    can start with Root entry {
        checker = check_new_emails();
        responder = draft_responses();
        here ++> checker;
        checker ++> responder;
        visit checker;
    }

    can check with check_new_emails entry {
        print("# Checking for new emails");
        self.abstracts = here.fetch_mail_abstracts(self.mailbox);
        if len(self.abstracts) == 0 {
            print("## No new emails");
            disengage;
        }
        print(f"## {len(self.abstracts)} new email threads");
        visit [-->];
    }

    can draft with draft_responses entry {
        for abstract in self.abstracts {
            category = here.filter_emails(abstract);
            print(f"### {abstract.thread_id} [{abstract.sender}] -> {category.value}");
            if category != Classification.IMPORTANT {
                continue;
            }
            thread = self.mailbox.get_thread(abstract.thread_id);
            if len(thread.emails) == 0 {
                continue;
            }
            analysis = here.email_action_agent(thread);
            if not isinstance(analysis, ThreadAnalysis) {
                error = unstructured_error(analysis, "email_action_agent", "ThreadAnalysis");
                self.mailbox.record_draft_error(str(analysis), error);
                print(f"### {abstract.thread_id}: {error}");
                continue;
            }
            reply = here.email_response_writer(analysis, thread);
            if not isinstance(reply, DraftReply) {
                # One retry, so a single formatting slip does not cost a whole
                # email; a second failure is recorded and the email skipped.
                print(f"### {abstract.thread_id}: writer returned unstructured output, retrying");
                reply = here.email_response_writer(analysis, thread);
            }
            if not isinstance(reply, DraftReply) {
                error = unstructured_error(reply, "email_response_writer", "DraftReply");
                self.mailbox.record_draft_error(str(reply), error);
                print(f"### {abstract.thread_id}: {error}");
                continue;
            }
            _ = self.mailbox.create_draft(reply.recipient, reply.subject, reply.message);
            self.drafted.append({"to": reply.recipient, "subject": reply.subject});
            print(f"### Draft created for {reply.recipient}: {reply.subject}");
        }
        report self.drafted;
    }
}


with entry:__main__ {
    dataset = os.environ.get("EMAIL_DATASET", "") or os.path.join(
        REPO_ROOT, "mock_mailbox", "datasets", "batch_001.json"
    );
    agent = EmailAgent(dataset_path=dataset);
    # Never lose a batch's captured drafts (or its token stats) to a crash
    # partway through the inbox -- results are written either way.
    failure = "";
    try {
        _ = root spawn agent;
    } except Exception as exc {
        failure = f"{type(exc).__name__}: {exc}";
        print(f"!! Walker aborted: {failure}");
    }

    out_path = agent.mailbox.save_results(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "mock_output",
        f"results_{agent.mailbox.case_id()}.json"
    ));
    print("=" * 60);
    print(f"Run complete for '{agent.mailbox.case_id()}'");
    print(f"Email threads passed to agents: {len(agent.abstracts)}");
    print(f"Drafts captured:                {len(agent.drafted)}");
    for d in agent.drafted {
        print(f"  - to: {d['to']} | subject: {d['subject']}");
    }
    print(f"LLM usage: {agent.mailbox.usage_line()}");
    print(f"Results saved to: {out_path}");
    if failure != "" {
        raise RuntimeError(failure);  # non-zero exit so a sweep cannot hide it
    }
}
