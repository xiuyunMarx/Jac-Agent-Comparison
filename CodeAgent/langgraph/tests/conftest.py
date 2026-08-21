"""Test fixtures, including the stand-in for byLLM's MockLLM.

`GenericFakeChatModel` from langchain-core cannot be used directly: it inherits
`BaseChatModel.bind_tools`, which raises NotImplementedError, so it can never
drive a tool-calling loop. `FakeToolCallingModel` below is the equivalent of
byLLM's `MockLLM(config={"outputs": [...]})` -- a scripted list of responses
consumed in order, with no network and no API key.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class MockToolCall:
    """One scripted tool call, the analogue of byLLM's MockToolCall."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)


class ScriptExhausted(AssertionError):
    pass


class FakeToolCallingModel(BaseChatModel):
    """A chat model that replays a scripted list of outputs.

    Each entry is one of:
      * `str`           -- a plain assistant reply, which ends the phase
      * `MockToolCall`  -- the assistant asks for one tool call
      * `list[MockToolCall]` -- the assistant asks for a parallel batch
      * `bool`          -- consumed by `with_structured_output`, i.e. the
                           objective_met classifier
    """

    outputs: list[Any] = []
    cursor: int = 0
    seen: list[list[BaseMessage]] = []
    # One entry per bind_tools call, in call order: what each phase was handed.
    bind_log: list[list[str]] = []

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def _next(self, what: str, wants_verdict: bool = False) -> Any:
        # Entries of the wrong kind are skipped rather than mis-delivered, so a
        # script may interleave phase replies with classifier verdicts the way
        # byLLM's MockLLM outputs list does.
        while self.cursor < len(self.outputs):
            value = self.outputs[self.cursor]
            self.cursor += 1
            if isinstance(value, bool) == wants_verdict:
                return value
        raise ScriptExhausted(
            f"the scripted model ran out of outputs while producing {what} "
            f"(consumed {self.cursor})"
        )

    def _to_message(self, value: Any) -> AIMessage:
        call_id = f"call_{self.cursor}"
        if isinstance(value, MockToolCall):
            value = [value]
        if isinstance(value, list) and value and isinstance(value[0], MockToolCall):
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": c.name, "args": dict(c.args), "id": f"{call_id}_{i}"}
                    for i, c in enumerate(value)
                ],
            )
        if isinstance(value, AIMessage):
            return value
        if isinstance(value, str):
            return AIMessage(content=value)
        raise ScriptExhausted(f"cannot turn {value!r} into an assistant message")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        message = self._to_message(self._next("an assistant message"))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        # The script drives every phase, so binding is a no-op beyond recording
        # which tools this phase was handed.
        self.bind_log.append(sorted(getattr(t, "name", str(t)) for t in tools))
        return self

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable:
        def call(_input: Any) -> Any:
            value = self._next("a structured verdict", wants_verdict=True)
            parsed = schema(met=value)
            if include_raw:
                return {"raw": AIMessage(content=""), "parsed": parsed, "parsing_error": None}
            return parsed

        return RunnableLambda(call)


def scripted(outputs: Sequence[Any]) -> FakeToolCallingModel:
    return FakeToolCallingModel(outputs=list(outputs), cursor=0, seen=[], bind_log=[])


@pytest.fixture
def tmp_repo() -> Iterator[str]:
    """A throwaway repository root, realpath'd so it compares equal to what the
    tools resolve internally (macOS /tmp is a symlink to /private/tmp)."""
    with tempfile.TemporaryDirectory() as path:
        yield os.path.realpath(path)


def write(root: str, rel: str, content: str) -> str:
    full = os.path.join(root, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    return full


def read(root: str, rel: str) -> str:
    with open(os.path.join(root, rel), "r", encoding="utf-8", newline="") as f:
        return f.read()
