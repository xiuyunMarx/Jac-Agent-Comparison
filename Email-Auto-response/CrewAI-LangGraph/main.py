"""Run the CrewAI-LangGraph email agent against the shared mock mailbox.

Usage:
    python main.py [path/to/dataset.json]

Defaults to ../mock_mailbox/datasets/batch_001.json. Requires OPENAI_API_KEY
in the environment or .env (the crew's LLM calls are real; the mailbox is
mocked). Captured drafts are written to mock_output/results_<case_id>.json.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))  # make the shared mock_mailbox package importable

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from mock_mailbox import MockMailbox


def main():
	dataset = sys.argv[1] if len(sys.argv) > 1 else str(
		ROOT.parent / "mock_mailbox" / "datasets" / "batch_001.json"
	)
	mailbox = MockMailbox(dataset)
	print(f"# Loaded mock mailbox '{mailbox.case_id}' ({len(mailbox.emails)} emails) from {dataset}")

	from src.graph import WorkFlow
	app = WorkFlow(mailbox).app
	final_state = app.invoke({})

	out_path = mailbox.save_results(
		ROOT / "mock_output" / f"results_{mailbox.case_id}.json", final_state
	)

	print("\n" + "=" * 60)
	print(f"Run complete for '{mailbox.case_id}'")
	print(f"Emails passed to crew: {len(final_state.get('emails', []))}")
	print(f"Drafts captured:       {len(mailbox.drafts)}")
	for d in mailbox.drafts:
		print(f"  - to: {d['to']} | subject: {d['subject']}")
	if mailbox.draft_errors:
		print(f"Draft tool errors:     {len(mailbox.draft_errors)}")
		for e in mailbox.draft_errors:
			print(f"  - {e['error']}")
	print(f"LLM usage:             {mailbox.usage_line()}")
	print(f"Results saved to:      {out_path}")


if __name__ == "__main__":
	main()
