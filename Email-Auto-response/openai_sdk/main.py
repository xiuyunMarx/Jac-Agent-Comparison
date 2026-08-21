"""Run the no-framework email agent against the shared mock mailbox.

Usage:
    python main.py [path/to/dataset.json]

Same CLI contract as the CrewAI-LangGraph baseline: one optional dataset path,
defaulting to ../mock_mailbox/datasets/batch_001.json ($EMAIL_DATASET, byLLM's
knob, is honoured too). Requires OPENAI_API_KEY in the environment or in a
.env next to this file (the LLM calls are real; the mailbox is mocked).
Captured drafts land in mock_output/results_<case_id>.json, written by the
shared MockMailbox.save_results() so eval/score.py reads them unchanged.

The run loop is byLLM's EmailAgent walker, written out: the mechanical inbox
scan, then classify -> analyze -> write per thread, with the walker's exact
skip/retry/record decisions at each stage. See nodes.py for the stages and
README.md for the mapping.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))  # make the shared mock_mailbox importable


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines into the environment, never overriding it.

    Hand-rolled so the dependency list stays exactly [openai] -- this is the
    no-framework side, and python-dotenv for six lines of parsing would be the
    first step back.
    """
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

from mock_mailbox import MockMailbox  # noqa: E402  (needs the sys.path line)

from nodes import (  # noqa: E402
    DraftReply,
    ThreadAnalysis,
    email_action_agent,
    email_response_writer,
    fetch_mail_abstracts,
    filter_emails,
    unstructured_error,
)


def run_agent(mailbox: MockMailbox) -> tuple[list, list[dict]]:
    """One pass over the batch. Returns (abstracts, drafted)."""
    owner = str(mailbox.owner_email or "")

    # -- check_new_emails: mechanical, no LLM --------------------------------
    print("# Checking for new emails")
    abstracts = fetch_mail_abstracts(mailbox)
    if not abstracts:
        print("## No new emails")
        return abstracts, []
    print(f"## {len(abstracts)} new email threads")

    # -- draft_responses: classify -> analyze -> write, per thread -----------
    drafted: list[dict] = []
    for abstract in abstracts:
        category = filter_emails(abstract, owner)
        print(f"### {abstract.thread_id} [{abstract.sender}] -> {category or '?'}")
        if category != "IMPORTANT":
            continue

        # Mechanical thread pull -- recorded by the harness for eval, exactly
        # as byLLM's MailBox.get_thread and CrewAI's Get Email Thread are.
        thread_text = str(mailbox.get_thread(abstract.thread_id))
        if thread_text.startswith("Error:"):
            # byLLM's `if len(thread.emails) == 0 { continue; }`.
            continue

        analysis = email_action_agent(thread_text, owner, mailbox)
        if not isinstance(analysis, ThreadAnalysis):
            error = unstructured_error(analysis, "email_action_agent", "ThreadAnalysis")
            mailbox.record_draft_error(str(analysis), error)
            print(f"### {abstract.thread_id}: {error}")
            continue

        reply = email_response_writer(analysis, thread_text, owner, mailbox)
        if not isinstance(reply, DraftReply):
            # One retry, so a single formatting slip does not cost a whole
            # email; a second failure is recorded and the email skipped.
            print(
                f"### {abstract.thread_id}: writer returned unstructured "
                "output, retrying"
            )
            reply = email_response_writer(analysis, thread_text, owner, mailbox)
        if not isinstance(reply, DraftReply):
            error = unstructured_error(reply, "email_response_writer", "DraftReply")
            mailbox.record_draft_error(str(reply), error)
            print(f"### {abstract.thread_id}: {error}")
            continue

        mailbox.create_draft(reply.recipient, reply.subject, reply.message)
        drafted.append({"to": reply.recipient, "subject": reply.subject})
        print(f"### Draft created for {reply.recipient}: {reply.subject}")
    return abstracts, drafted


def main() -> int:
    dataset = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("EMAIL_DATASET", "")
        or str(ROOT.parent / "mock_mailbox" / "datasets" / "batch_001.json")
    )
    # The mailbox is constructed before any LLM client exists: its constructor
    # installs the shared token meter, which must be patched into the openai
    # SDK ahead of the first request or the run records no usage.
    mailbox = MockMailbox(dataset)
    print(
        f"# Loaded mock mailbox '{mailbox.case_id}' "
        f"({len(mailbox.emails)} emails) from {dataset}"
    )

    # Never lose a batch's captured drafts (or its token stats) to a crash
    # partway through the inbox -- results are written either way.
    abstracts, drafted, failure = [], [], ""
    try:
        abstracts, drafted = run_agent(mailbox)
    except Exception as exc:  # noqa: BLE001 - reported, saved, then re-raised
        failure = f"{type(exc).__name__}: {exc}"
        print(f"!! Run aborted: {failure}")

    out_path = mailbox.save_results(
        ROOT / "mock_output" / f"results_{mailbox.case_id}.json"
    )
    print("=" * 60)
    print(f"Run complete for '{mailbox.case_id}'")
    print(f"Email threads passed to agents: {len(abstracts)}")
    print(f"Drafts captured:                {len(drafted)}")
    for d in drafted:
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
