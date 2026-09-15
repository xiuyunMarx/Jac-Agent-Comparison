"""The meeting-assistant pipeline in NVIDIA Object-Oriented Agents (NOOA).

../byLLM/nodes.jac, written the way NOOA's docs say to write an agent: an
agent is a Python object, an async method ending in `...` is implemented by
the LLM, a method with a real body is ordinary Python, and the workflow is a
hidden deterministic method. No graph, no walker.

    node GeneratingTasks            -> class GeneratingTasks(Agent)
      analyse_meeting_transcript by llm  ->   async def analyse_meeting_transcript(...) -> MeetingTasks: ...
      repair_meeting_tasks by llm        ->   async def repair_meeting_tasks(...) -> MeetingTasks: ...
      (walker: extract, validate, repair, validate-or-raise)
                                    ->   @hidden async def extract(...): the evidence gate, in Python
    obj MeetingTask + sem strings   -> Pydantic MeetingTask with Field(description=...)
    node AddTask / Save2CSV / SendNotification (no model calls)
                                    -> one-method classes, as on the openai_sdk side
    walker MeetingAssistant         -> class MeetingAssistant, run(): extract, then the fan-out in insertion order

Model seam: $BENCH_MODEL (bare id, default glm-5.2) becomes litellm's
openai/<id> against $OPENAI_BASE_URL with $OPENAI_API_KEY, the same knobs
every arm reads. byLLM's defaults go on the wire: temperature 0.7, no
max_tokens. Both agentic methods use PredictStrategy (one structured call,
validated against the return type, up to three attempts: byLLM's
max_output_retries=3), not NOOA's default CodeAct REPL loop; the Jac side
hands the model no tools, so there is nothing for CodeAct to do.
"""

from __future__ import annotations

import os
from typing import Annotated, Any

from pydantic import BaseModel, Field

from nooa import Agent, EventQuery, hidden, strategy
from nooa.config import PredictConfig
from nooa.config.truncation_config import FormatConfig, TruncationConfig
from nooa.strategies import PredictStrategy
from nooa.unifiedllm import CompletionClient, LLMResponse

from tools import (
    create_trello_card,
    save_tasks_to_csv,
    send_message_to_channel,
    token_usage,
)

# ---------------------------------------------------------------------------
# Configuration: the Jac file's `glob` block.
# ---------------------------------------------------------------------------

# One knob for every arm: $BENCH_MODEL is the bare model id (default glm-5.2);
# NOOA is litellm-routed, so it gets the "openai/" provider prefix and the
# OpenAI-compatible endpoint from $OPENAI_BASE_URL (ollama.com by default).
BENCH_MODEL: str = os.environ.get("BENCH_MODEL", "glm-5.2")
MODEL_NAME: str = BENCH_MODEL if "/" in BENCH_MODEL else f"openai/{BENCH_MODEL}"
MODEL_BASE: str = os.environ.get("OPENAI_BASE_URL", "https://ollama.com/v1")
API_KEY: Annotated[str, hidden] = os.environ.get("OPENAI_API_KEY", "")
# byLLM's default: jac.toml sets no [byllm.call_params], and the byllm runtime
# then sends temperature=0.7 on every call. No max_tokens, matching every side.
TEMPERATURE: float = 0.7
# gpt-5 / o-series accept only the default temperature and bill their thinking
# against the completion budget; "minimal" switches that off. Absent elsewhere.
REASONING_MODEL: bool = MODEL_NAME.split("/", 1)[-1].startswith(("gpt-5", "o1", "o3", "o4"))
# byLLM's `by llm(max_output_retries=3)`: attempts at a valid typed reply.
OUTPUT_ATTEMPTS: int = 3


# ---------------------------------------------------------------------------
# Typed results: `obj MeetingTask` with its `sem` strings as Field descriptions.
# NOOA renders them into the schema the model sees and validates against.
# ---------------------------------------------------------------------------


class MeetingTask(BaseModel):
    name: str = Field(description="Required, non-empty, short, actionable title for the task")
    description: str = Field(
        description=(
            "Required, non-empty, detailed description of the task: clear instructions, owner, "
            "deadline, steps to reproduce, and acceptance criteria where applicable"
        )
    )


class MeetingTasks(BaseModel):
    """The extracted action items. A named model rather than a bare list[MeetingTask]: NOOA wraps
    a bare list in a one-field model whose key only appears in the response_format schema, which
    ollama.com ignores for GLM; the class rendered in the prompt is what the model reads."""

    tasks: list[MeetingTask] = Field(
        default_factory=list,
        description="Every concrete action item in the transcript; empty only when there are none.",
    )


def _tasks_are_valid(tasks: list[MeetingTask]) -> bool:
    for task in tasks:
        if not task.name.strip() or not task.description.strip():
            return False
    return True


# ---------------------------------------------------------------------------
# The model seam: NOOA's litellm CompletionClient, subclassed only to feed the
# token counters from each response at the call site.
# ---------------------------------------------------------------------------


class TracedClient(CompletionClient):
    async def acall(self, messages: list[dict[str, Any]], *args: Any, **kwargs: Any) -> LLMResponse:
        response = await super().acall(messages, *args, **kwargs)
        token_usage.track(response.usage)
        return response


LLM_CALL_PARAMS: dict[str, Any] = {"reasoning_effort": "minimal"} if REASONING_MODEL else {"temperature": TEMPERATURE}
llm = TracedClient(
    model=MODEL_NAME,
    api_base=MODEL_BASE,
    api_key=API_KEY or "ollama",
    drop_params=True,       # get_llm_client()'s default: litellm drops what a model rejects
    **LLM_CALL_PARAMS,
)


def by_llm() -> PredictStrategy:
    """`by llm(max_output_retries=3)`: one structured call, validated, up to three attempts."""
    return PredictStrategy(config=PredictConfig(max_retries=OUTPUT_ATTEMPTS))


# Arguments are rendered whole: a long transcript must reach the model uncut
# (NOOA's default cuts rendered argument strings at 2000 chars).
UNTRUNCATED = TruncationConfig(prefill_format=FormatConfig(max_string=None, max_length=None, max_depth=None))


# ---------------------------------------------------------------------------
# node GeneratingTasks
# ---------------------------------------------------------------------------


class GeneratingTasks(Agent, llm=llm, event_query=EventQuery.current_call(), truncation=UNTRUNCATED):
    """Turns a meeting transcript into the list of concrete action items it contains."""

    @strategy(by_llm())
    async def analyse_meeting_transcript(
        self,
        transcript: Annotated[str, "The complete meeting transcript from which to extract action items"],
    ) -> MeetingTasks:
        """Extract every concrete action item from the meeting transcript. Return a MeetingTasks
        object whose tasks are MeetingTask objects. Every object must contain exactly two non-empty string fields: name
        and description; never return an empty or placeholder task. Keep distinct commitments for
        different deliverables or owners as separate tasks. Combine discussion steps that belong to
        the same deliverable into one task, including approval followed by execution, rather than
        over-splitting them. Return an empty list only when the transcript contains no action
        items. Document owners, deadlines, requirements, and acceptance criteria stated in the
        transcript without inventing facts."""
        ...

    @strategy(by_llm())
    async def repair_meeting_tasks(
        self,
        transcript: Annotated[str, "The complete meeting transcript, which is the sole source of truth"],
        invalid_output: Annotated[str, "The rejected task list containing empty or malformed fields"],
    ) -> MeetingTasks:
        """The previous extraction contained one or more empty or malformed tasks. Re-extract every
        concrete action item from the transcript and return a corrected JSON list of MeetingTask
        objects. Every object must contain exactly two non-empty string fields: name and
        description; never return an empty or placeholder task. Keep distinct commitments for
        different deliverables or owners as separate tasks. Combine steps belonging to the same
        deliverable into one task rather than over-splitting them. Return an empty list only when
        the transcript contains no action items. Use the invalid output only to understand what
        must be corrected, not as a source of facts."""
        ...

    @hidden
    async def extract(self, transcript: str) -> list[MeetingTask]:
        """The walker's `generate_task` ability, in Python: extract, check, repair once, check
        again or fail. The gate is deterministic and unskippable; the model supplies judgment."""
        tasks = (await self.analyse_meeting_transcript(transcript)).tasks
        if not _tasks_are_valid(tasks):
            print("Generated task list contains empty fields. Retrying once with corrective feedback...")
            tasks = (await self.repair_meeting_tasks(transcript, str(tasks))).tasks
        if not _tasks_are_valid(tasks):
            raise ValueError("Meeting task extraction still contains empty fields after corrective retry")
        return tasks


# ---------------------------------------------------------------------------
# The fan-out nodes: no model calls, so plain Python classes (the NOOA docs:
# a class with no agentic method does not subclass Agent).
# ---------------------------------------------------------------------------


class AddTask:
    def work(self, tasks: list[MeetingTask]) -> None:
        for task in tasks:
            if task.name and task.description:
                create_trello_card(task.name, task.description)
            else:
                print("Task is missing a name or description. Skipping...")


class Save2CSV:
    def work(self, tasks: list[MeetingTask]) -> None:
        save_tasks_to_csv([(task.name, task.description) for task in tasks])


class SendNotification:
    def work(self, tasks: list[MeetingTask]) -> None:
        send_message_to_channel(f"{len(tasks)} New tasks have been added to Trello!")


# ---------------------------------------------------------------------------
# walker MeetingAssistant: a coordinator with no agentic method, so an
# ordinary class that owns the agent object and sequences the fan-out.
# ---------------------------------------------------------------------------


class MeetingAssistant:
    def __init__(self, transcript: str, generator: GeneratingTasks | None = None) -> None:
        self.transcript = transcript
        self.tasks: list[MeetingTask] = []
        self.generator = generator or GeneratingTasks()

    async def run(self) -> "MeetingAssistant":
        self.tasks = await self.generator.extract(self.transcript)
        # byLLM's insertion order: AddTask, Save2CSV, SendNotification.
        for node in (AddTask(), Save2CSV(), SendNotification()):
            node.work(self.tasks)
        return self
