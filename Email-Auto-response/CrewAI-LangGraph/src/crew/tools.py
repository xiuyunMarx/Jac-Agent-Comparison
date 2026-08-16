from crewai.tools import tool


def build_tools(mailbox):
	"""CrewAI tools backed by the shared MockMailbox."""

	@tool("Get Email Thread")
	def get_thread(thread_id: str) -> str:
		"""Pull the complete email thread for a given thread ID.
		The input should be the thread ID string only, e.g. `thr_001`."""
		return mailbox.get_thread(thread_id)

	@tool("Search the internet")
	def web_search(query: str) -> str:
		"""Search the internet for information about a topic and return relevant results."""
		return mailbox.web_search(query)

	@tool("Create Draft")
	def create_draft(data: str) -> str:
		"""
		Useful to create an email draft.
		The input to this tool should be a pipe (|) separated text
		of length 3 (three), representing who to send the email to,
		the subject of the email and the actual message.
		For example, `lorem@ipsum.com|Nice To Meet You|Hey it was great to meet you.`.
		"""
		# Same parsing as the original Gmail CreateDraftTool, so inputs with
		# extra pipes fail the same way they would against real Gmail.
		try:
			email, subject, message = data.split('|')
		except ValueError as exc:
			mailbox.record_draft_error(data, str(exc))
			raise
		result = mailbox.create_draft(email, subject, message)
		return f"\nDraft created: {result}\n"

	return get_thread, web_search, create_draft
