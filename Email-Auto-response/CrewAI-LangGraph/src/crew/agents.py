import os
from textwrap import dedent
from crewai import Agent, LLM

from .tools import build_tools

# $BENCH_MODEL (bare id, default glm-5.2) in litellm's shape, on the
# OpenAI-compatible endpoint from $OPENAI_BASE_URL.
_BENCH_MODEL = os.environ.get("BENCH_MODEL", "glm-5.2")
MODEL_LLM = LLM(
    model=_BENCH_MODEL if "/" in _BENCH_MODEL else f"openai/{_BENCH_MODEL}",
    base_url=os.environ.get("OPENAI_BASE_URL", "https://ollama.com/v1"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    # byLLM's default; CrewAI sends no temperature at all unless asked, which
    # left this arm on the provider default while the other two ran at 0.7.
    # Omitted for gpt-5 / o-series, which accept only the default.
    **({"reasoning_effort": "minimal"} if _BENCH_MODEL.split("/", 1)[-1].startswith(("gpt-5", "o1", "o3", "o4")) else {"temperature": 0.7}),
)

class EmailFilterAgents():
	def __init__(self, mailbox):
		self.get_thread, self.web_search, self.create_draft = build_tools(mailbox)

	def email_filter_agent(self):
		return Agent(
			llm=MODEL_LLM,
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
			llm=MODEL_LLM,
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
			llm=MODEL_LLM,
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
