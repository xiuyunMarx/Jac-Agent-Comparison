from typing import Any

from crewai import LLM, Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from meeting_assistant_flow.types import (
    MeetingTaskList,
)


@CrewBase
class MeetingAssistantCrew:
    """Meeting Assistant Crew"""

    # Populated by @CrewBase from config/agents.yaml and config/tasks.yaml
    # (its default paths) at instantiation time.
    agents_config: dict[str, Any]
    tasks_config: dict[str, Any]
    agents: list[BaseAgent]
    tasks: list[Task]
    llm = LLM(model="gpt-4o")

    @agent
    def meeting_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config["meeting_analyzer"],
            llm=self.llm,
        )

    @task
    def analyze_meeting(self) -> Task:
        # description/expected_output are filled in from tasks.yaml via config
        return Task(  # pyright: ignore[reportCallIssue]
            config=self.tasks_config["analyze_meeting"],
            output_pydantic=MeetingTaskList,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Meeting Issue Generation Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
