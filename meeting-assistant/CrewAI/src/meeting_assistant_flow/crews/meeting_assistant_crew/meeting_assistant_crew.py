import os
from typing import Any

from crewai import LLM, Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from meeting_assistant_flow.types import (
    MeetingTaskList,
)


def _model_name() -> str:
    """$BENCH_MODEL (bare id, default glm-5.2) in litellm's shape: openai/<id>."""
    name = os.environ.get("BENCH_MODEL", "glm-5.2")
    return name if "/" in name else f"openai/{name}"


@CrewBase
class MeetingAssistantCrew:
    """Meeting Assistant Crew"""

    # Populated by @CrewBase from config/agents.yaml and config/tasks.yaml
    # (its default paths) at instantiation time.
    agents_config: dict[str, Any]
    tasks_config: dict[str, Any]
    agents: list[BaseAgent]
    tasks: list[Task]
    # Hard-coded to the same model as byLLM (nodes.jac) and openai_sdk (nodes.py).
    llm = LLM(
        model=_model_name(),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://ollama.com/v1"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

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
