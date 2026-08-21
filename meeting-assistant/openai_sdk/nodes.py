"""The meeting-assistant pipeline: one extraction call, then a deterministic fan-out.

This mirrors ../byLLM/nodes.jac node for node. What that side gets from the
framework -- the prompt assembled from `sem` strings, the strict JSON schema
derived from `list[MeetingTask]`, the parse back into objects, litellm's
request plumbing -- is written out here by hand: SYSTEM_PROMPT / USER_PROMPT
carry the same words the sem strings do, RESPONSE_FORMAT is the schema
byLLM's type_to_schema would derive, and _parse_tasks() is the decode step.

The class names are byLLM's node names (GeneratingTasks, AddTask, Save2CSV,
SendNotification) and MeetingAssistant.run() is the walker's traversal, so
the two files diff cleanly.
"""

import json
import os
from dataclasses import dataclass

from tools import (
    create_trello_card,
    save_tasks_to_csv,
    send_message_to_channel,
    token_usage,
)

# Pinned identically on all three sides of the comparison, or the benchmark
# measures the model rather than the framework: $MEETING_MODEL is the one knob
# all three read. byLLM and CrewAI need litellm's "openai/" prefix on an
# unfamiliar name; the raw SDK wants the bare id, so a prefix is stripped rather
# than passed through. Unset, every side stays on gpt-4o.
MODEL = os.environ.get("MEETING_MODEL", "").replace(":", "/").split("/")[-1] or "gpt-4o"
# byLLM's default: its jac.toml sets no [byllm.call_params], and the byllm
# runtime then sends temperature=0.7 on every call. CrewAI's LLM(model=
# "gpt-4o") sends none at all (provider default 1.0); the fidelity target is
# byLLM, so 0.7 is what goes on the wire. No max_tokens, matching both sides.
TEMPERATURE = 0.7

_client = None


def _get_client():
    # Imported lazily so `import nodes` needs neither the openai package nor
    # an API key -- the pipeline shape can be inspected without credentials.
    global _client
    if _client is None:
        from openai import OpenAI

        # OPENAI_API_KEY and OPENAI_BASE_URL are read natively by the SDK.
        _client = OpenAI()
    return _client


@dataclass
class MeetingTask:
    """One extracted action item. byLLM's `obj MeetingTask`, fields and all."""

    name: str = ""
    description: str = ""


# ---------------------------------------------------------------------------
# The prompts. This is what the other two sides express through framework
# constructs; every sentence is traceable (see README.md, "The prompts"):
#   - the role paragraph is CrewAI's agents.yaml backstory;
#   - the instruction paragraph is byLLM's `sem GeneratingTasks.
#     analyse_meeting_transcript` verbatim;
#   - the two field rules are byLLM's `sem MeetingTask.name` and
#     `sem MeetingTask.description` verbatim (CrewAI's tasks.yaml
#     expected_output says the same thing in one sentence);
#   - the user message frames the transcript with CrewAI's own words.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a meeting transcript analysis agent. You are an expert in "
    "analyzing meeting transcripts and summarizing the discussions into "
    "actionable tasks. Your ability to identify important issues helps "
    "ensure teams can follow up and address key points effectively.\n"
    "\n"
    "Analyze the meeting transcript and break the discussion down into a "
    "list of important, well-structured, actionable tasks that a team can "
    "follow up on. Document each task thoroughly.\n"
    "\n"
    "Every task has exactly two fields:\n"
    "- name: Short, actionable title for the task.\n"
    "- description: Detailed description of the task: clear instructions, "
    "steps to reproduce, and acceptance criteria where applicable.\n"
    "\n"
    "Reply with a JSON object of the form "
    '{"tasks": [{"name": ..., "description": ...}, ...]}.'
)

USER_PROMPT = "Here is the meeting transcript for your reference:\n\n{transcript}"

# What the model is told when its reply did not decode: one corrective retry,
# the same recovery class byLLM's runtime performs on a schema miss.
RETRY_PROMPT = (
    "That reply could not be read as a task list ({error}). Reply again with "
    'only a JSON object of the form {{"tasks": [{{"name": ..., '
    '"description": ...}}, ...]}}.'
)

# The schema byLLM derives from `-> list[MeetingTask]` plus the field sems:
# strict json_schema, the list wrapped in a single-key object (OpenAI requires
# an object root; byLLM's wrapper key is its internal "schema_object_wrapper",
# here it is "tasks" -- CrewAI's MeetingTaskList spelling -- which the model
# sees but the scoring never does).
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "MeetingTaskList",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Short, actionable title for the task",
                            },
                            "description": {
                                "type": "string",
                                "description": (
                                    "Detailed description of the task: clear "
                                    "instructions, steps to reproduce, and "
                                    "acceptance criteria where applicable"
                                ),
                            },
                        },
                        "required": ["name", "description"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["tasks"],
            "additionalProperties": False,
        },
    },
}


def _complete(messages: list[dict]) -> str:
    """One round trip, recorded. The whole of the model seam."""
    response = _get_client().chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=messages,
        response_format=RESPONSE_FORMAT,
    )
    token_usage.track(response)
    return response.choices[0].message.content or ""


def _parse_tasks(reply: str) -> list[MeetingTask]:
    """Decode the model's JSON into MeetingTask objects, or raise."""
    data = json.loads(reply)
    items = data.get("tasks") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("no task list in the reply")
    tasks = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"task entry is not an object: {item!r}")
        tasks.append(
            MeetingTask(
                name=str(item.get("name", "")),
                description=str(item.get("description", "")),
            )
        )
    return tasks


class GeneratingTasks:
    """byLLM's `def analyse_meeting_transcript(...) by llm()`, written out."""

    def analyse_meeting_transcript(self, transcript: str) -> list[MeetingTask]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(transcript=transcript)},
        ]
        reply = _complete(messages)
        try:
            return _parse_tasks(reply)
        except (ValueError, json.JSONDecodeError) as exc:
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": RETRY_PROMPT.format(error=exc)},
            ]
            return _parse_tasks(_complete(messages))


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


class MeetingAssistant:
    """byLLM's `walker MeetingAssistant`; run() is the traversal.

    The walker visits GeneratingTasks first and then its three children in
    insertion order -- AddTask, Save2CSV, SendNotification -- each handed the
    extracted task list. A for loop over that order is the same walk.
    """

    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.tasks: list[MeetingTask] = []

    def run(self) -> "MeetingAssistant":
        self.tasks = GeneratingTasks().analyse_meeting_transcript(self.transcript)
        for node in (AddTask(), Save2CSV(), SendNotification()):
            node.work(self.tasks)
        return self
