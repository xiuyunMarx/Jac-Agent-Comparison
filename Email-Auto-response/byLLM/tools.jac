"""Typed Jac wrapper around the shared MockMailbox (../mock_mailbox).

Agent functions consume Email/Thread objects instead of the raw text the
CrewAI tools work with; drafts and thread requests are still recorded in the
underlying harness for eval.
"""
import os;
import sys;

glob REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
glob _PATH_READY: bool = (REPO_ROOT in sys.path) or (sys.path.insert(0, REPO_ROOT) is None);

import from mock_mailbox { MockMailbox }


obj Email {
    has subject: str;
    has body: str;
    has sender: str;
    has thread_id: str;
}

obj Thread {
    has thread_id: str;
    has subject: str;
    has sender: str;
    has recipient: str;
    has emails: list[Email];
}

obj MailBox {
    has dataset_path: str;
    has impl: any by postinit;

    def postinit {
        self.impl = MockMailbox(self.dataset_path);
    }

    # read all emails in the mailbox
    def search() -> list[dict] {
        return [dict(x) for x in self.impl.search()];
    }

    def get_thread(thread_id: str) -> Thread {
        self.impl.get_thread(thread_id);  # records the request for eval
        tid = str(thread_id).strip().strip("'\"");
        for e in self.impl.emails {
            if e["threadId"] == tid {
                return Thread(
                    thread_id=tid,
                    subject=str(e.get("subject", "")),
                    sender=str(e["sender"]),
                    recipient=str(self.impl.owner_email),
                    emails=[
                        Email(
                            subject=str(e.get("subject", "")),
                            body=str(m["body"]),
                            sender=str(m["from"]),
                            thread_id=tid
                        ) for m in e["full_thread"]
                    ]
                );
            }
        }
        return Thread(thread_id=tid, subject="", sender="", recipient="", emails=[]);
    }

    def web_search(query: str) -> str {
        return str(self.impl.web_search(query));
    }

    def create_draft(recipient: str, subject: str, message: str) -> str {
        result = self.impl.create_draft(recipient, subject, message);
        return str(result);
    }

    def owner_email() -> str {
        return str(self.impl.owner_email);
    }

    # A stage that answered in prose instead of the declared type is recorded
    # the same way a malformed Create-Draft call is, so eval sees it.
    def record_draft_error(raw_input: str, error: str) {
        self.impl.record_draft_error(raw_input, error);
    }

    def case_id() -> str {
        return str(self.impl.case_id);
    }

    # Token/cost totals for the LLM calls made during this run; the harness
    # meters them at the openai-SDK level, so byllm needs no instrumentation.
    def usage_line() -> str {
        return str(self.impl.usage_line());
    }

    def save_results(path: str) -> str {
        return str(self.impl.save_results(path));
    }
}

with entry:__main__ {
    mb = MailBox(dataset_path=os.path.join(REPO_ROOT, "mock_mailbox", "datasets", "batch_001.json"));
    print(len(mb.search()), "emails in batch");
    t = mb.get_thread("thr_001");
    print("thread", t.thread_id, "from", t.sender, "-", len(t.emails), "messages");
    print("usage:", mb.usage_line());  # zero calls here: no LLM ran
}
