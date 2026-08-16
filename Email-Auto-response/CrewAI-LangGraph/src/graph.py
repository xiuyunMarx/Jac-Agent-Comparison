from langgraph.graph import StateGraph, END

from .state import EmailsState
from .nodes import Nodes
from .crew.crew import EmailFilterCrew

class WorkFlow():
	"""Single-pass workflow over one mock-mailbox batch.

	The original Gmail version looped forever (wait 180s -> poll again); with a
	static mock batch there is nothing new to poll, so the graph ends after one
	check_new_emails -> draft_responses pass.
	"""

	def __init__(self, mailbox):
		nodes = Nodes(mailbox)
		crew = EmailFilterCrew(mailbox)
		workflow = StateGraph(EmailsState)

		workflow.add_node("check_new_emails", nodes.check_email)
		workflow.add_node("draft_responses", crew.kickoff)

		workflow.set_entry_point("check_new_emails")
		workflow.add_conditional_edges(
				"check_new_emails",
				nodes.new_emails,
				{
					"continue": 'draft_responses',
					"end": END
				}
		)
		workflow.add_edge('draft_responses', END)
		self.app = workflow.compile()
