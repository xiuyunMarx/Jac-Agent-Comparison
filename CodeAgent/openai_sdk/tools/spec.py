"""The tool surface, written in the terms the OpenAI API actually takes.

This is the whole of what LangChain's `StructuredTool` + pydantic argument model
was doing for the other Python side: turn a bound method plus some prose into
`{"type": "function", "function": {...}}` and then call it with a dict of
arguments the model produced.

byLLM derives the same JSON Schema from `sem` strings attached to the archetype;
LangGraph derives it from `Field(description=...)` on a `BaseModel`. Here it is
written out, so all three sides send the model the same words and the same
parameter shapes -- which is the premise of the comparison, not a detail of it.

The schemas produced here are byte-identical to what
`convert_to_openai_tool(StructuredTool.from_function(...))` produces on the
LangGraph side, `required` list included; `openai_sdk/README.md` records how to
check that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

JSON = dict[str, Any]

# `default=None` cannot mean "no default": a parameter may legitimately default
# to None, to 0, or to "" -- and `expected_count=1`/`start_line=1` would then be
# indistinguishable from required parameters.
_NO_DEFAULT: Any = object()


def prop(
    type_: str,
    description: str,
    *,
    default: Any = _NO_DEFAULT,
    enum: Iterable[Any] | None = None,
    items: JSON | None = None,
) -> JSON:
    """One parameter's schema. Carrying a `default` is what makes it optional."""
    spec: JSON = {"type": type_, "description": description}
    if enum is not None:
        spec["enum"] = list(enum)
    if items is not None:
        spec["items"] = items
    if default is not _NO_DEFAULT:
        spec["default"] = default
    return spec


def schema(**properties: JSON) -> JSON:
    """An object schema over `properties`, required list derived from defaults.

    `required` is omitted entirely when nothing is required, rather than emitted
    empty -- that is what pydantic's `model_json_schema()` does, and show_plan is
    the tool that shows the difference.
    """
    out: JSON = {"type": "object", "properties": dict(properties)}
    required = [name for name, spec in properties.items() if "default" not in spec]
    if required:
        out["required"] = required
    return out


def _coerce(name: str, value: Any, spec: JSON) -> tuple[Any, str]:
    """Bend `value` to the declared type, or explain why it will not go.

    Returns (value, "") on success and (None, "Error: ...") on failure. The
    coercions here are the ones pydantic performs in its default lax mode on the
    LangGraph side, and no others: a model that answers `start_line: "12"`
    is served identically by both, and one that answers `content: 12` is
    refused by both.
    """
    want = spec.get("type")
    if want == "integer":
        # bool is an int in Python and nothing else is; `expected_count: true`
        # is a mistake worth naming rather than silently reading as 1.
        if isinstance(value, bool):
            return None, f"Error: '{name}' must be an integer, not a boolean."
        if isinstance(value, int):
            return value, ""
        if isinstance(value, float) and value.is_integer():
            return int(value), ""
        if isinstance(value, str):
            try:
                return int(value.strip()), ""
            except ValueError:
                pass
        return None, f"Error: '{name}' must be an integer; got {value!r}."
    if want == "string":
        if isinstance(value, str):
            return value, ""
        return None, f"Error: '{name}' must be a string; got {type(value).__name__}."
    if want == "array":
        if not isinstance(value, (list, tuple)):
            return None, f"Error: '{name}' must be a list; got {type(value).__name__}."
        item_type = (spec.get("items") or {}).get("type")
        if item_type == "string" and any(not isinstance(v, str) for v in value):
            return None, f"Error: every entry of '{name}' must be a string."
        return list(value), ""
    return value, ""


@dataclass(frozen=True)
class Tool:
    """One capability, its documentation, and the function behind it."""

    name: str
    description: str
    parameters: JSON
    fn: Callable[..., str]

    def openai_spec(self) -> JSON:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def invoke(self, args: Mapping[str, Any] | None) -> str:
        """Call the tool with model-supplied arguments. Never raises.

        Every failure comes back as an `Error: ` string naming what to do
        differently, because that is the only channel the model has. A raised
        exception would end the phase, where byLLM and LangGraph both lose one
        tool call and carry on -- and the phase's summary is what the next phase
        reads, so ending it early costs the run, not the call.
        """
        args = dict(args or {})
        props: JSON = self.parameters.get("properties") or {}
        required: list[str] = list(self.parameters.get("required") or [])

        unknown = [k for k in args if k not in props]
        if unknown:
            allowed = ", ".join(props) or "(none)"
            return (
                f"Error: {self.name} has no parameter "
                f"{', '.join(repr(u) for u in sorted(unknown))}. "
                f"Its parameters are: {allowed}."
            )
        missing = [k for k in required if k not in args]
        if missing:
            return (
                f"Error: {self.name} requires "
                f"{', '.join(repr(m) for m in missing)}, which was not supplied."
            )
        kwargs: JSON = {}
        for key, value in args.items():
            coerced, refusal = _coerce(key, value, props[key])
            if refusal:
                return refusal
            kwargs[key] = coerced
        try:
            out = self.fn(**kwargs)
        except Exception as e:  # noqa: BLE001 - a tool must not end the phase
            return f"Error: {self.name} failed: {type(e).__name__}: {e}"
        # The str-only contract described in tools/common.py, enforced at the
        # one boundary every tool result passes through.
        if not isinstance(out, str):
            return str(out)
        return out or f"Error: {self.name} returned nothing."


def tool_specs(tools: Iterable[Tool]) -> list[JSON]:
    return [t.openai_spec() for t in tools]


def by_name(tools: Iterable[Tool]) -> dict[str, Tool]:
    return {t.name: t for t in tools}


__all__ = [
    "JSON",
    "Tool",
    "by_name",
    "prop",
    "schema",
    "tool_specs",
]
