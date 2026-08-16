from textwrap import dedent
from crewai import Agent

from .tools import build_tools

class EmailFilterAgents():
	def __init__(self, mailbox):
		self.get_thread, self.web_search, self.create_draft = build_tools(mailbox)

	def email_filter_agent(self):
		return Agent(
			role='Senior Email Analyst',
			goal='Filter out non-essential emails like newsletters and promotional content',
			backstory=dedent("""\
				As a Senior Email Analyst, you have extensive experience in email content analysis.
				You are adept at distinguishing important emails from spam, newsletters, and other
				irrelevant content. Your expertise lies in identifying key patterns and markers that
				signify the importance of an email."""),
			verbose=True,
			allow_delegation=False
		)

	def email_action_agent(self):
		return Agent(
			role='Email Action Specialist',
			goal='Identify action-required emails and compile a list of their IDs',
			backstory=dedent("""\
				With a keen eye for detail and a knack for understanding context, you specialize
				in identifying emails that require immediate action. Your skill set includes interpreting
				the urgency and importance of an email based on its content and context."""),
			tools=[
				self.get_thread,
				self.web_search
			],
			verbose=True,
			allow_delegation=False,
		)

	def email_response_writer(self):
		return Agent(
			role='Email Response Writer',
			goal='Draft responses to action-required emails',
			backstory=dedent("""\
				You are a skilled writer, adept at crafting clear, concise, and effective email responses.
				Your strength lies in your ability to communicate effectively, ensuring that each response is
				tailored to address the specific needs and context of the email."""),
			tools=[
				self.web_search,
				self.get_thread,
				self.create_draft
			],
			verbose=True,
			allow_delegation=False,
		)
