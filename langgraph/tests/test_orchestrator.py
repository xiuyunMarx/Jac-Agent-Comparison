"""Port of byLLM/tests/orchestrator_tests.jac.

The byLLM suite asserts the capability boundary by traversing `Exposes` edges
and the topology by traversing `Flow` edges. Here the boundary is
`Phase.exposes` and the topology is `FLOW`, and both are additionally checked
against the *compiled* LangGraph -- so an edge declared in FLOW but never wired
into the StateGraph would still fail.
"""

from __future__ import annotations

import os
import time

import pytest
from conftest import MockToolCall, ScriptExhausted, scripted
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

import orchestrator as orch
from orchestrator import (
    FLOW,
    WORKING_PHASES,
    build_app,
    build_pipeline,
    make_phase_node,
    rt,
    set_model,
)
from phase_agent import (
    MAX_REACT_ITERATIONS,
    MAX_TOOL_RESULT_CHARS,
    UNANSWERED_TOOL_CALL,
    SerialToolNode,
    batch_signature,
    build_phase_agent,
    progress_guard,
    run_phase,
)
from telemetry import TokenUsage


@pytest.fixture(autouse=True)
def _restore_model():
    yield
    set_model(None)


# ---------------------------------------------------------------------------
# Graph shape. These need no model at all: the capability boundary is data, so
# it can be asserted by reading it.
# ---------------------------------------------------------------------------


def test_build_pipeline_starts_at_planning() -> None:
    phases = build_pipeline(rt)
    assert "Planning" in phases
    assert phases["Planning"].title == "Planning"
    assert WORKING_PHASES[0] == "Planning"


def test_only_editing_may_write_and_only_editing_and_verifying_may_run() -> None:
    phases = build_pipeline(rt)

    plan_tools = phases["Planning"].tool_names()
    assert "read_file" in plan_tools
    assert "grep" in plan_tools
    assert "set_plan" in plan_tools
    assert "write_file" not in plan_tools
    assert "run_command" not in plan_tools

    assert "write_file" not in phases["Exploring"].tool_names()
    assert "run_command" not in phases["Exploring"].tool_names()
    # set_plan belongs to Planning alone; the other phases only update it.
    assert "set_plan" not in phases["Exploring"].tool_names()

    edit_tools = phases["Editing"].tool_names()
    assert "write_file" in edit_tools
    assert "replace_in_file" in edit_tools
    assert "run_command" in edit_tools
    assert "read_file" in edit_tools

    verify_tools = phases["Verifying"].tool_names()
    assert "run_command" in verify_tools
    assert "write_file" not in verify_tools


def test_every_stalling_phase_has_an_escape_edge_to_finished() -> None:
    # Every phase that can stall has an escape edge to the terminal node, so a
    # spent step budget always lands on Finished instead of the run falling off
    # the end of the graph with no answer.
    assert "Finished" in FLOW["Planning"]
    assert "Finished" in FLOW["Exploring"]
    assert "Finished" in FLOW["Verifying"]
    # The repair loop back to Editing.
    assert "Editing" in FLOW["Verifying"]
    # Editing never shortcuts past verification.
    assert "Finished" not in FLOW["Editing"]
    assert FLOW["Editing"] == ("Verifying",)


def test_the_compiled_graph_wires_exactly_the_declared_topology() -> None:
    set_model(scripted([]))
    graph = build_app().get_graph()
    nodes = set(graph.nodes)
    for title in WORKING_PHASES:
        assert title in nodes
    assert "Finished" in nodes

    wired: dict[str, set[str]] = {}
    for edge in graph.edges:
        wired.setdefault(edge.source, set()).add(edge.target)
    for source, dests in FLOW.items():
        for dest in dests:
            assert dest in wired.get(source, set()), f"{source} -> {dest} not wired"
    # ... and nothing beyond it: a router can only reach a declared destination.
    for title in WORKING_PHASES:
        assert wired[title] == set(FLOW[title])


def test_exposed_tools_are_categorised_data() -> None:
    phases = build_pipeline(rt)
    planning = phases["Planning"].categories()
    assert "read" in planning
    assert "track" in planning
    assert "write" not in planning

    editing = phases["Editing"].categories()
    assert "write" in editing
    assert "run" in editing


def test_every_phase_tool_is_uniquely_named() -> None:
    # A duplicate name silently shadows a tool in the bound schema.
    for phase in build_pipeline(rt).values():
        names = phase.tool_names()
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Progress guard.
# ---------------------------------------------------------------------------


def test_the_progress_guard_aborts_on_three_identical_calls() -> None:
    state = {"recent_sigs": []}
    for _ in range(2):
        state["recent_sigs"].append("grep|No matches")
        assert progress_guard(state) == "model"
    # Third identical call in a row ends the phase rather than burning budget.
    state["recent_sigs"].append("grep|No matches")
    assert progress_guard(state) == "summarize"


def test_the_progress_guard_tolerates_alternating_calls() -> None:
    state = {"recent_sigs": []}
    for sig in ["grep|x", "read_file|y", "grep|x", "read_file|y"]:
        state["recent_sigs"].append(sig)
        assert progress_guard(state) == "model"


def test_the_batch_signature_folds_a_whole_turn() -> None:
    # byLLM's IterationContext carries one (tool, result) pair; a LangGraph turn
    # can carry a batch, so the whole batch has to fold into one signature or
    # two different batches would look alike.
    a = batch_signature([
        ToolMessage(content="hit", name="grep", tool_call_id="1"),
        ToolMessage(content="body", name="read_file", tool_call_id="2"),
    ])
    b = batch_signature([ToolMessage(content="hit", name="grep", tool_call_id="1")])
    assert a != b
    assert a == batch_signature([
        ToolMessage(content="hit", name="grep", tool_call_id="1"),
        ToolMessage(content="body", name="read_file", tool_call_id="2"),
    ])


# ---------------------------------------------------------------------------
# The tool-dispatch node: serialization and result clipping.
# ---------------------------------------------------------------------------


def _batch(*calls: tuple[str, dict]) -> dict:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": name, "args": args, "id": f"c{i}"}
                    for i, (name, args) in enumerate(calls)
                ],
            )
        ],
        "recent_sigs": [],
    }


def test_a_batch_containing_a_mutating_call_never_interleaves() -> None:
    # byLLM marks the mutating tools with mark_serialize; LangGraph's ToolNode
    # dispatches a batch through executor.map, so without this a write and a
    # run_command issued in one turn would race and the run would stop being
    # reproducible.
    events: list[str] = []

    def slow(tag: str) -> str:
        events.append(f"enter:{tag}")
        time.sleep(0.05)
        events.append(f"exit:{tag}")
        return f"OK: {tag}"

    node = SerialToolNode([
        StructuredTool.from_function(
            func=lambda file_path, content: slow(file_path),
            name="write_file",
            description="write",
        ),
        StructuredTool.from_function(
            func=lambda command: slow(command),
            name="run_command",
            description="run",
        ),
    ])
    node(_batch(("write_file", {"file_path": "a", "content": "x"}),
                ("run_command", {"command": "b"})))
    assert events == ["enter:a", "exit:a", "enter:b", "exit:b"]


def test_a_serialized_batch_still_answers_every_call() -> None:
    node = SerialToolNode([
        StructuredTool.from_function(
            func=lambda file_path, content: f"OK: wrote {file_path}",
            name="write_file",
            description="write",
        ),
        StructuredTool.from_function(
            func=lambda command: f"$ {command}",
            name="run_command",
            description="run",
        ),
    ])
    out = node(_batch(("write_file", {"file_path": "a", "content": "x"}),
                      ("run_command", {"command": "git status"})))
    produced = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.name for m in produced] == ["write_file", "run_command"]
    assert len(out["recent_sigs"]) == 1


def test_tool_results_are_clipped_to_the_phase_budget() -> None:
    node = SerialToolNode([
        StructuredTool.from_function(
            func=lambda file_path: "y" * 9000,
            name="read_file",
            description="read",
        )
    ])
    out = node(_batch(("read_file", {"file_path": "big.txt"})))
    content = out["messages"][0].content
    assert len(content) < 9000
    assert "truncated" in content
    assert content.startswith("y" * MAX_TOOL_RESULT_CHARS)


# ---------------------------------------------------------------------------
# The two loop brakes, driven through the compiled phase agent. byLLM gets both
# from `max_react_iterations` and `on_iteration`; here they are edges, so they
# are worth exercising end to end and not just as predicates.
# ---------------------------------------------------------------------------


def _exploring_agent(model, tmp_repo: str):
    rt.retarget(tmp_repo)
    phase = build_pipeline(rt)["Exploring"]
    return build_phase_agent(phase.tools(), lambda: model, TokenUsage()), phase


def test_a_repeated_call_ends_the_phase_with_a_summary(tmp_repo: str) -> None:
    repeat = MockToolCall("grep", {"pattern": "absent", "path": "."})
    model = scripted([repeat, repeat, repeat, "I kept getting the same answer."])
    agent, phase = _exploring_agent(model, tmp_repo)

    summary, messages = run_phase(agent, [SystemMessage(content="s"), HumanMessage(content=phase.goal)])

    assert summary == "I kept getting the same answer."
    # Three identical results, then the abort -- not a fourth attempt.
    assert len([m for m in messages if isinstance(m, ToolMessage)]) == 3
    assert model.cursor == len(model.outputs)


def test_the_iteration_cap_ends_the_phase_with_a_summary(tmp_repo: str) -> None:
    # Distinct patterns so each result differs and the repeat guard stays quiet:
    # this must be the iteration cap firing, not the progress guard.
    calls = [
        MockToolCall("grep", {"pattern": f"absent{i}", "path": "."})
        for i in range(MAX_REACT_ITERATIONS)
    ]
    model = scripted([*calls, "I ran out of room."])
    agent, phase = _exploring_agent(model, tmp_repo)

    summary, messages = run_phase(agent, [SystemMessage(content="s"), HumanMessage(content=phase.goal)])

    assert summary == "I ran out of room."
    # Exactly MAX_REACT_ITERATIONS model calls, and the last one's tool request
    # is answered with a placeholder rather than left dangling -- an unanswered
    # tool call is an invalid conversation for the provider.
    assert model.cursor == MAX_REACT_ITERATIONS + 1
    assert any(
        isinstance(m, ToolMessage) and m.content == UNANSWERED_TOOL_CALL
        for m in messages
    )


# ---------------------------------------------------------------------------
# Full traversals driven by a scripted model -- no API key, no network.
# ---------------------------------------------------------------------------


def test_a_full_traversal_edits_a_real_file(tmp_repo: str) -> None:
    with open(os.path.join(tmp_repo, "greet.py"), "w") as f:
        f.write('print("hello")\n')

    set_model(scripted([
        # Planning
        MockToolCall("set_plan", {"steps": ["change the greeting"]}),
        "Plan written: change the greeting.",
        # Exploring
        MockToolCall("grep", {"pattern": "hello", "path": "."}),
        "greet.py line 1 holds the greeting.",
        # Editing
        MockToolCall("replace_in_file", {"file_path": "greet.py", "old": "hello", "new": "goodbye"}),
        "Replaced hello with goodbye in greet.py.",
        # Verifying
        MockToolCall("run_command", {"command": "git status"}),
        "git status ran; the file is modified.",
        # objective_met
        True,
    ]))

    result = orch.solve("change the greeting to goodbye", tmp_repo, max_steps=10)

    # The run reached the terminal node and carried an answer out.
    assert result.answer != ""
    assert result.steps == 4
    for phase in ["Planning", "Exploring", "Editing", "Verifying"]:
        assert phase in result.answer
    # The edit really happened on disk -- the tools were called, not described.
    with open(os.path.join(tmp_repo, "greet.py")) as f:
        assert "goodbye" in f.read()
    # The tool log is what the eval harness reads.
    assert [c.name for c in result.tool_calls] == [
        "set_plan", "grep", "replace_in_file", "run_command"
    ]
    assert result.llm_calls == 9


def test_the_repair_loop_sends_failed_verification_back_to_editing(tmp_repo: str) -> None:
    with open(os.path.join(tmp_repo, "greet.py"), "w") as f:
        f.write('print("hello")\n')

    set_model(scripted([
        "planned",           # Planning
        "explored",          # Exploring
        "first attempt",     # Editing
        "tests failed",      # Verifying
        False,               # objective_met -> back to Editing
        "second attempt",    # Editing
        "tests passed",      # Verifying
        True,                # objective_met -> Finished
    ]))

    result = orch.solve("fix the failing test", tmp_repo, max_steps=10)
    assert result.steps == 6
    assert result.answer.count("### Editing") == 2
    assert result.answer.count("### Verifying") == 2
    assert result.answer.endswith("tests passed")


def test_the_step_cap_bounds_a_never_satisfied_run(tmp_repo: str) -> None:
    # The step cap bounds the run even when the model never says it is done.
    with open(os.path.join(tmp_repo, "x.txt"), "w") as f:
        f.write("body\n")

    outputs: list[object] = []
    # Enough phase replies for the cap to bite, each followed by "not done".
    for i in range(40):
        outputs.append(f"phase reply {i}")
        outputs.append(False)
    set_model(scripted(outputs))

    result = orch.solve("never satisfied", tmp_repo, max_steps=6)
    assert result.answer != ""
    # It stops at the cap rather than looping Editing <-> Verifying forever.
    assert result.steps <= 8


def test_a_read_only_request_can_finish_without_editing(tmp_repo: str) -> None:
    # "what does this function do" is answered, not edited. Planning owns the
    # shortcut, which only opens once the step budget is spent.
    with open(os.path.join(tmp_repo, "x.txt"), "w") as f:
        f.write("body\n")
    set_model(scripted(["it does nothing"]))

    result = orch.solve("what does x.txt say", tmp_repo, max_steps=1)
    assert result.steps == 1
    assert "### Planning" in result.answer
    assert "### Editing" not in result.answer


def test_each_phase_is_bound_to_only_its_own_tools(tmp_repo: str) -> None:
    # The capability boundary has to survive into what the model is actually
    # handed, not merely be declared in the phase table.
    model = scripted(["planned", "explored", "edited", "verified", True])
    set_model(model)
    orch.solve("anything", tmp_repo, max_steps=10)

    # One bind_tools call per phase, in WORKING_PHASES order.
    assert len(model.bind_log) == len(WORKING_PHASES)
    planning, exploring, editing, verifying = model.bind_log
    assert "set_plan" in planning and "write_file" not in planning
    assert "write_file" not in exploring and "run_command" not in exploring
    assert "write_file" in editing and "run_command" in editing
    assert "run_command" in verifying and "write_file" not in verifying


def test_a_script_that_runs_dry_fails_loudly(tmp_repo: str) -> None:
    # Guards the tests themselves: a short script must not quietly pass by
    # ending the run early.
    set_model(scripted(["only one reply"]))
    with pytest.raises(ScriptExhausted):
        orch.solve("anything", tmp_repo, max_steps=10)


def test_entering_a_phase_scopes_the_explorers_served_window_memory(tmp_repo: str) -> None:
    # The explorer refuses a window it has already served *in this phase*. That
    # is only correct if the phase node stamps the scope on the way in -- byLLM
    # does it in `Phase.work`, and without the equivalent here Exploring's reads
    # would suppress Editing's read-back after an edit.
    rt.retarget(tmp_repo)
    phase = build_pipeline(rt)["Editing"]

    class _StubAgent:
        def invoke(self, _state, _config=None):
            return {"messages": [AIMessage(content="done")]}

    node = make_phase_node(phase, rt, _StubAgent())
    rt.explorer.set_scope("Exploring")
    node({
        "objective": "o", "max_steps": 4, "steps": 0, "ledger": [],
        "answer": "", "edits_made": False, "last_summary": "", "phase_msgs": {},
    })
    assert rt.explorer.scope == "Editing"
