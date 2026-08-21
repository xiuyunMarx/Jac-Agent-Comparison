"""The per-phase ReAct loop, written out.

byLLM gets the whole of this from one clause:

    def run_phase(directive: str) -> str by agent_model(
        tools=rt.phase_tools, conversation=rt.phase_ctx, temperature=0.0,
        max_tokens=4096, max_react_iterations=25, on_iteration=progress_guard,
        max_tool_result_length=MAX_TOOL_RESULT_CHARS
    );

LangGraph gets most of it from a compiled StateGraph. Here it is a `while` loop,
reproducing the five behaviours that clause specifies:

  * a hard 25-iteration cap;
  * abort-with-summary when the model repeats the identical call three times
    running, or reaches for the same tool six times with nothing in between;
  * tool results clipped to the tools' own budget;
  * mutating tool calls that never run concurrently;
  * temperature 0 and a 4096-token completion ceiling (in llm.py).

Both brakes and the cap end the phase by asking the model to summarise, never by
raising. A phase that ends with no text contributes nothing to the ledger, and
the next phase then works blind.

One thing this file does *not* have to do: serialize the tool batch. byLLM needs
`mark_serialize` and LangGraph needs a whole `SerialToolNode` subclass, because
both dispatch a parallel batch concurrently by default and two concurrent writes
make a run unreproducible -- fatal for an A/B benchmark. Sequential execution is
what a `for` loop already is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from llm import assistant_turn, complete, tool_turn
from telemetry import TokenUsage
from tools.common import MAX_TOOL_CHARS, clip
from tools.spec import Tool, by_name, tool_specs

MAX_REACT_ITERATIONS: int = 25
# Tied to the tools' own budget rather than set independently, and deliberately
# not set below it. Every tool already truncates its result, and does so in the
# terms the model can act on -- read_file stops on a line boundary and names the
# start_line that resumes it. A smaller cap here would cut that result a second
# time, mid-line and with no continuation hint, and the model has no way to tell
# a silently shortened result from a complete one: the only move a mystery
# truncation leaves is to repeat the identical call.
MAX_TOOL_RESULT_CHARS: int = MAX_TOOL_CHARS
REPEAT_ABORT_THRESHOLD: int = 3
# One tool called over and over with nothing else between it is the other way a
# phase stalls, and the identical-call test cannot see it: read_file walking a
# file one line at a time issues a different call every time, so every signature
# differs. In one measured astropy run five of the eight phase laps died this
# way -- 109 of 162 LLM calls and 75% of the run's prompt tokens spent paging
# through a file the phase had no further use for. Six is above any honest run
# of paging that grep could not have replaced, and the abort ends the phase with
# a summary, so a false positive costs one phase rather than the run.
SAME_TOOL_ABORT_THRESHOLD: int = 6
# How much of a tool result the repeat signature compares. byLLM hands its
# on_iteration hook a 500-character last_result, so comparing the full
# MAX_TOOL_RESULT_CHARS here would make this brake far less eager than byLLM's
# for the same conversation.
SIG_RESULT_CHARS: int = 500

# byLLM folds its `sem run_phase` strings into the prompt; the same words go in
# here, so the three sides give the model the same instruction.
PHASE_TASK = (
    "Carry out the current phase of the coding task using the tools provided. "
    "Work until the phase's goal(Directive) is met, then reply with a short "
    "summary of what you did and what the next phase needs to know."
)
DIRECTIVE_LABEL = "Directive -- what to accomplish in this phase, which is the phase's goal"

ABORT_DIRECTIVE = (
    "Stop calling tools now. Reply with a short summary of what you did in this "
    "phase and what the next phase needs to know."
)

UNANSWERED_TOOL_CALL = "(not run: the phase stopped before this call was reached)"

# What ended the phase, for the caller and for the fallback summary. "" means the
# model stopped on its own, which is the ordinary case.
ABORT_REASONS: dict[str, str] = {
    "repeat": (
        "the phase was stopped after issuing the identical tool call "
        f"{REPEAT_ABORT_THRESHOLD} times in a row"
    ),
    "same_tool": (
        "the phase was stopped after reaching for the same tool "
        f"{SAME_TOOL_ABORT_THRESHOLD} times with nothing in between"
    ),
    "cap": (
        f"the phase reached its {MAX_REACT_ITERATIONS}-iteration limit"
    ),
}


@dataclass
class PhaseResult:
    """What one phase visit produced."""

    summary: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    # "" | "repeat" | "same_tool" | "cap"
    aborted: str = ""


def all_alike(history: Sequence[str], n: int) -> bool:
    """True when the last `n` entries of `history` are all the same value."""
    if len(history) < n:
        return False
    tail = history[len(history) - n:]
    return all(item == tail[0] for item in tail)


def progress_guard(recent_sigs: Sequence[str], recent_tools: Sequence[str]) -> str:
    """Why the phase should stop, or "" to continue.

    byLLM's `on_iteration` hook, kept as a function of the two histories so it
    can be reasoned about on its own. No legitimate workflow issues the identical
    call three times running, and none reaches for the same tool six times with
    nothing in between.
    """
    if all_alike(recent_sigs, REPEAT_ABORT_THRESHOLD):
        return "repeat"
    if all_alike(recent_tools, SAME_TOOL_ABORT_THRESHOLD):
        return "same_tool"
    return ""


def parse_arguments(raw: str) -> tuple[dict[str, Any], str]:
    """The tool call's arguments, or ({}, "Error: ...") explaining the refusal."""
    if not raw or not raw.strip():
        return {}, ""
    try:
        parsed = json.loads(raw)
    except ValueError as e:
        return {}, (
            f"Error: the arguments were not valid JSON ({e}). Re-issue the call "
            "with a well-formed JSON object."
        )
    if not isinstance(parsed, dict):
        return {}, (
            "Error: the arguments must be a JSON object of named parameters, "
            f"not {type(parsed).__name__}."
        )
    return parsed, ""


def dispatch(registry: Mapping[str, Tool], name: str, raw_args: str) -> str:
    """Run one tool call. Never raises; every failure is a string the model reads."""
    tool = registry.get(name)
    if tool is None:
        # The phase is only offered its own tools, so this is a hallucinated
        # name rather than a boundary crossing -- but naming the boundary is
        # what stops the model trying the same unavailable tool again.
        available = ", ".join(sorted(registry)) or "(none)"
        return (
            f"BLOCKED: there is no tool named '{name}' in this phase. "
            f"The tools available here are: {available}."
        )
    args, refusal = parse_arguments(raw_args)
    if refusal:
        return refusal
    return tool.invoke(args)


def last_text(messages: Sequence[Mapping[str, Any]]) -> str:
    """The final thing the model actually said, for the phase ledger."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def opening_messages(
    conversation: Sequence[Mapping[str, Any]], directive: str
) -> list[dict[str, Any]]:
    """The phase's starting messages: its system prompt, then what to do."""
    return list(conversation) + [
        {
            "role": "user",
            "content": f"{PHASE_TASK}\n\n{DIRECTIVE_LABEL}: {directive}",
        }
    ]


def run_phase(
    directive: str,
    tools: Sequence[Tool],
    conversation: Sequence[Mapping[str, Any]],
    *,
    usage: TokenUsage | None = None,
    client: Any | None = None,
    model: str | None = None,
) -> PhaseResult:
    """Drive one phase to completion and return its summary and conversation."""
    registry = by_name(tools)
    specs = tool_specs(tools)
    messages = opening_messages(conversation, directive)
    recent_sigs: list[str] = []
    recent_tools: list[str] = []
    iterations = 0

    while True:
        message = complete(
            messages, tools=specs, usage=usage, client=client, model=model
        )
        messages.append(assistant_turn(message))
        iterations += 1

        calls = list(getattr(message, "tool_calls", None) or [])
        if not calls:
            return PhaseResult(
                summary=last_text(messages) or _fallback(""),
                messages=messages,
                iterations=iterations,
            )

        # The model still wants tools but the budget is gone. Let it summarise
        # rather than cutting the phase off mid-thought.
        if iterations >= MAX_REACT_ITERATIONS:
            for call in calls:
                messages.append(
                    tool_turn(call.id, call.function.name, UNANSWERED_TOOL_CALL)
                )
            return _summarize(
                messages, "cap", iterations, usage=usage, client=client, model=model
            )

        # One call at a time, in the order the model asked for them.
        brake = ""
        for call in calls:
            name = call.function.name
            if brake:
                # A brake mid-batch: the remaining calls are answered but not
                # run, both because the conversation is invalid for the provider
                # with an unanswered tool call in it, and because "you are
                # looping" is exactly the moment not to execute the next write.
                messages.append(tool_turn(call.id, name, UNANSWERED_TOOL_CALL))
                continue
            result = clip(dispatch(registry, name, call.function.arguments),
                          MAX_TOOL_RESULT_CHARS)
            messages.append(tool_turn(call.id, name, result))
            # byLLM's IterationContext carries one (last_tool, last_result) pair
            # per iteration; recorded per executed call here, which is the same
            # thing whenever a turn asks for one tool and finer-grained when it
            # asks for several.
            recent_sigs.append(f"{name}|{result[:SIG_RESULT_CHARS]}")
            recent_tools.append(name)
            brake = progress_guard(recent_sigs, recent_tools)

        if brake:
            return _summarize(
                messages, brake, iterations, usage=usage, client=client, model=model
            )


def _summarize(
    messages: list[dict[str, Any]],
    reason: str,
    iterations: int,
    *,
    usage: TokenUsage | None,
    client: Any | None,
    model: str | None,
) -> PhaseResult:
    """byLLM's IterationAction.ABORT_WITH_SUMMARY.

    Asked without tools: offering them here just invites another call, which is
    the behaviour that made the abort necessary.
    """
    messages.append({"role": "user", "content": ABORT_DIRECTIVE})
    message = complete(messages, tools=None, usage=usage, client=client, model=model)
    messages.append(assistant_turn(message))
    return PhaseResult(
        summary=last_text(messages) or _fallback(reason),
        messages=messages,
        iterations=iterations,
        aborted=reason,
    )


def _fallback(reason: str) -> str:
    """What goes in the ledger when the model produced no text at all.

    Never "": the ledger is the whole of what the next phase and the final
    answer are built from, and a blank entry is indistinguishable from a phase
    that ran and found nothing.
    """
    why = ABORT_REASONS.get(reason, "the phase produced no closing summary")
    return f"(no summary: {why}.)"


__all__ = [
    "ABORT_DIRECTIVE",
    "MAX_REACT_ITERATIONS",
    "MAX_TOOL_RESULT_CHARS",
    "REPEAT_ABORT_THRESHOLD",
    "SAME_TOOL_ABORT_THRESHOLD",
    "PhaseResult",
    "all_alike",
    "dispatch",
    "last_text",
    "opening_messages",
    "parse_arguments",
    "progress_guard",
    "run_phase",
]
