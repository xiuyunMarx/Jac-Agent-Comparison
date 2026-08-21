"""Task state: the agent's own plan, written down and read back.

The only tool group here with no effect outside the process. Its whole
mechanism is that structured state written by the model is rendered back into
the transcript, so intent survives a long tool-call sequence.

Port of byLLM/nodes/plan.jac. The Jac `TaskStatus` enum existed to put a
JSON-Schema `enum` constraint on the parameter, and LangGraph spelled that as a
`Literal[...]` pydantic renders into one; with no framework in between it is
written as the `enum` it always was.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tools.common import log_tool_call
from tools.spec import Tool, prop, schema

TaskStatus = Literal["todo", "doing", "done", "blocked"]
# The same four values, in the form the JSON Schema needs them.
TASK_STATUSES: tuple[str, ...] = ("todo", "doing", "done", "blocked")


@dataclass
class PlanItem:
    id: int
    title: str
    status: str = "todo"
    note: str = ""


SET_PLAN_PARAMS = schema(
    steps=prop(
        "array",
        "The steps to take, in order, each a short imperative sentence.",
        items={"type": "string"},
    ),
)

UPDATE_TASK_PARAMS = schema(
    task_id=prop(
        "integer", "The number of the step to update, as shown in the plan."
    ),
    status=prop("string", "The step's new status.", enum=TASK_STATUSES),
    note=prop(
        "string",
        "An optional short note, for example why a step is blocked. Pass an "
        "empty string to leave the existing note unchanged.",
        default="",
    ),
)

SHOW_PLAN_PARAMS = schema()


SET_PLAN_DOC = (
    "Replace the whole plan with a new ordered list of steps, numbered from 1. "
    "Call this once you know how you intend to solve the task, and again if the "
    "approach changes materially. Returns the rendered plan."
)

UPDATE_TASK_DOC = (
    "Change the status of one step, optionally attaching a short note. Mark a "
    "step 'doing' when you start it and 'done' once you have verified it. "
    "Returns the rendered plan."
)

SHOW_PLAN_DOC = (
    "Show the current plan with each step's status. Use this to re-orient after "
    "a long stretch of work."
)


class PlanTasks:
    """Keeps the agent's plan: an ordered checklist of the steps needed to finish."""

    def __init__(self) -> None:
        self.items: list[PlanItem] = []
        self.next_id: int = 1

    def set_plan(self, steps: list[str]) -> str:
        log_tool_call("set_plan", {"steps": str(len(steps))})
        cleaned = [s.strip() for s in steps if s.strip()]
        if not cleaned:
            return "Error: 'steps' was empty. Pass at least one step."
        self.items = []
        self.next_id = 1
        for title in cleaned:
            self.items.append(PlanItem(id=self.next_id, title=title))
            self.next_id += 1
        return f"OK: plan set with {len(self.items)} steps.\n" + self._render()

    def update_task(self, task_id: int, status: str, note: str = "") -> str:
        log_tool_call("update_task", {"task_id": str(task_id), "status": str(status)})
        for item in self.items:
            if item.id == task_id:
                item.status = str(status)
                if note:
                    item.note = note
                return f"OK: step {task_id} is now {item.status}.\n" + self._render()
        if not self.items:
            return (
                f"Error: there is no step {task_id} because no plan has been set. "
                "Call set_plan first."
            )
        known = ", ".join(str(i.id) for i in self.items)
        return f"Error: no step with id {task_id}. Known ids: {known}."

    def show_plan(self) -> str:
        log_tool_call("show_plan", {})
        return self._render()

    def _render(self) -> str:
        if not self.items:
            return "(no plan yet -- call set_plan with the steps you intend to take)"
        marks = {"todo": "[ ]", "doing": "[~]", "done": "[x]", "blocked": "[!]"}
        lines: list[str] = []
        done = 0
        for item in self.items:
            if item.status == "done":
                done += 1
            mark = marks.get(item.status, "[ ]")
            line = f"{mark} {item.id}. {item.title}"
            if item.note:
                line += f"  -- {item.note}"
            lines.append(line)
        return "\n".join(lines) + f"\n({done}/{len(self.items)} done)"

    def as_tools(self) -> list[Tool]:
        """Every tool this holder owns. Phases pick from these individually --
        Planning is the only phase granted set_plan."""
        return [self.set_plan_tool(), self.update_task_tool(), self.show_plan_tool()]

    def set_plan_tool(self) -> Tool:
        return Tool(
            name="set_plan",
            description=SET_PLAN_DOC,
            parameters=SET_PLAN_PARAMS,
            fn=self.set_plan,
        )

    def update_task_tool(self) -> Tool:
        return Tool(
            name="update_task",
            description=UPDATE_TASK_DOC,
            parameters=UPDATE_TASK_PARAMS,
            fn=self.update_task,
        )

    def show_plan_tool(self) -> Tool:
        return Tool(
            name="show_plan",
            description=SHOW_PLAN_DOC,
            parameters=SHOW_PLAN_PARAMS,
            fn=self.show_plan,
        )
