"""Framework-agnostic mock Gmail mailbox for evaluating email auto-response agents.

Loads a dataset JSON (one batch of emails with full threads and ground-truth
labels) and exposes the three operations the agents need:

  - search()                  -> snippet-level results, like GmailSearch
  - get_thread(thread_id)     -> full thread text, like GmailGetThread
  - create_draft(to, subj, m) -> captures the draft instead of touching Gmail

Captured drafts (and failed draft attempts) are the primary evaluation
artifact; save_results() writes them to a JSON file.

Constructing a mailbox also installs the shared token meter (see
token_meter.py), so every implementation's LLM token usage and dollar cost is
captured with no agent-side changes and lands in the same results JSON.
"""

import json
from pathlib import Path

from .token_meter import install_token_meter


class MockMailbox:
    def __init__(self, dataset_path):
        self.dataset_path = str(dataset_path)
        # Counters start at zero per mailbox, i.e. per run over one batch.
        self.token_meter = install_token_meter(reset=True)
        with open(dataset_path) as f:
            data = json.load(f)
        self.case_id = data.get("case_id", Path(dataset_path).stem)
        self.owner_email = data.get("owner_email", "")
        self.emails = data["emails"]
        self._threads = {e["threadId"]: e for e in self.emails}
        self._search_results = data.get("web_search_results", {})
        self.drafts = []
        self.draft_errors = []
        self.thread_requests = []
        self.web_queries = []

    # -- Gmail-like surface ------------------------------------------------

    def search(self, query=None):
        """Snippet-level view of the inbox, shaped like GmailSearch output."""
        return [
            {
                "id": e["id"],
                "threadId": e["threadId"],
                "snippet": e["snippet"],
                "sender": e["sender"],
                "subject": e.get("subject", ""),
            }
            for e in self.emails
        ]

    def get_thread(self, thread_id):
        """Full thread as readable text, shaped like GmailGetThread output."""
        thread_id = str(thread_id).strip().strip("'\"")
        self.thread_requests.append(thread_id)
        email = self._threads.get(thread_id)
        if email is None:
            return (
                f"Error: no thread found with ID '{thread_id}'. "
                f"Valid thread IDs are: {', '.join(self._threads)}"
            )
        lines = [f"Thread ID: {thread_id}", f"Subject: {email.get('subject', '')}", ""]
        for msg in email["full_thread"]:
            lines += [
                f"From: {msg['from']}",
                f"To: {msg['to']}",
                f"Date: {msg.get('date', '')}",
                "",
                msg["body"],
                "-" * 40,
            ]
        return "\n".join(lines)

    def web_search(self, query):
        """Canned web search: deterministic stand-in for Tavily."""
        self.web_queries.append(query)
        for keyword, result in self._search_results.items():
            if keyword.lower() in query.lower():
                return result
        return "No relevant results found for this query."

    def create_draft(self, to, subject, message):
        """Capture a draft instead of creating it in Gmail."""
        draft = {"to": to.strip(), "subject": subject.strip(), "message": message}
        self.drafts.append(draft)
        return {"to": [draft["to"]], "subject": draft["subject"], "status": "draft saved (mock)"}

    def record_draft_error(self, raw_input, error):
        self.draft_errors.append({"raw_input": raw_input, "error": error})

    # -- Evaluation output -------------------------------------------------

    def usage_summary(self, include_calls=True):
        """Token/cost totals for the LLM calls made during this run."""
        return self.token_meter.summary(include_calls=include_calls)

    def usage_line(self):
        """One-line token/cost summary, for a runner to print."""
        return self.token_meter.format_line()

    def save_results(self, path, final_state=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "case_id": self.case_id,
            "dataset": self.dataset_path,
            "owner_email": self.owner_email,
            "drafts": self.drafts,
            "draft_errors": self.draft_errors,
            "thread_requests": self.thread_requests,
            "web_queries": self.web_queries,
            "usage": self.usage_summary(),
        }
        if final_state is not None:
            results["crew_result"] = str(final_state.get("action_required_emails", ""))
            results["emails_passed_to_crew"] = final_state.get("emails", [])
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        return path
