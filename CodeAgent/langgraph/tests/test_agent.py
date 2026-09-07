"""Exercise the real agent runtime with scripted HTTP responses; no API key."""

import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
import nodes
import orchestrator


def answer(text):
    return {"role": "assistant", "content": text}


def calls(*items):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"call_{i}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}
        for i, (name, args) in enumerate(items)
    ]}


@contextmanager
def model_script(*replies):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        message = replies[len(requests) - 1]
        return httpx.Response(200, json={
            "id": f"reply_{len(requests)}", "object": "chat.completion", "created": 0,
            "model": "test", "choices": [{"index": 0, "message": message,
                                         "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
                      "prompt_tokens_details": {"cached_tokens": 3}},
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatOpenAI(model="test", api_key="test", http_client=client,
                           max_retries=0, use_responses_api=False)
        with patch.object(main, "get_model", return_value=model):
            yield requests


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        previous = nodes.repo.base_dir
        self.addCleanup(setattr, nodes.repo, "base_dir", previous)
        nodes.repo.base_dir = self.temp.name
        Path(self.temp.name, "calc.py").write_text("def add(a, b):\n    return a - b\n")
        nodes.history.clear()
        nodes.tool_log.clear()

    def test_native_tool_schemas_keep_defaults_and_descriptions(self):
        tool = next(t for t in nodes.PHASES["Plan"].toolbox() if t.name == "read_file")
        schema = tool.args_schema.model_json_schema()
        self.assertEqual(schema["required"], ["path"])
        self.assertEqual(schema["properties"]["start_line"]["default"], 1)
        self.assertEqual(schema["properties"]["path"]["description"], "Path relative to the repository root.")
        self.assertIn("return a - b", tool.invoke({"path": "calc.py"}))

    def test_phase_history_and_tool_permissions(self):
        with model_script(calls(("read_file", {"path": "calc.py"})), answer("plan"),
                          answer("explore")) as requests:
            self.assertEqual(main.work(nodes.PHASES["Plan"]), "plan")
            self.assertEqual(main.work(nodes.PHASES["Explore"]), "explore")
        for request, name in [(requests[0], "Plan"), (requests[2], "Explore")]:
            self.assertEqual([t["function"]["name"] for t in request["tools"]], list(nodes.PHASES[name].tools))
            self.assertIn(nodes.PHASES[name].goal, request["messages"][-1]["content"])
        self.assertTrue(any(m["role"] == "tool" for m in requests[2]["messages"]))
        self.assertTrue(any(m.get("content") == "plan" for m in requests[2]["messages"]))

    def test_phase_limit_requests_one_tool_free_summary(self):
        read = calls(("read_file", {"path": "calc.py"}))
        with model_script(read, read, answer("summary")) as requests:
            self.assertEqual(main.work(replace(nodes.PHASES["Plan"], max_react_iterations=2)), "summary")
        self.assertEqual(len(nodes.tool_log), 2)
        self.assertNotIn("tools", requests[-1])
        self.assertEqual(requests[-1]["messages"][-1]["content"], main.FINAL_INSTRUCTION)
        self.assertFalse(any(m.content == main.FINAL_INSTRUCTION for m in nodes.history))

    def test_shared_budget_is_checked_between_rounds_including_first_round_exception(self):
        for initial in (59, 60):
            with self.subTest(initial=initial):
                nodes.history.clear()
                nodes.tool_log[:] = [{"name": "read_file", "ok": True}] * initial
                batch = calls(("read_file", {"path": "calc.py"}), ("outline", {"path": "calc.py"}))
                with model_script(batch, answer("budget reached")) as requests:
                    main.work(nodes.PHASES["Plan"])
                self.assertEqual(len(nodes.tool_log), initial + 2)
                self.assertEqual([c["name"] for c in nodes.tool_log[-2:]], ["read_file", "outline"])
                self.assertNotIn("tools", requests[-1])

    def test_invalid_tool_arguments_are_returned_to_model(self):
        with model_script(calls(("read_file", {"path": "calc.py", "start_line": "invalid"})),
                          answer("recovered")) as requests:
            self.assertEqual(main.work(nodes.PHASES["Plan"]), "recovered")
        self.assertIn("Error", requests[-1]["messages"][-1]["content"])

    def test_tool_execution_errors_are_recoverable(self):
        with patch.object(nodes.repo, "_abs", side_effect=OSError("unreadable")):
            with model_script(calls(("read_file", {"path": "calc.py"})), answer("recovered")) as requests:
                self.assertEqual(main.work(nodes.PHASES["Plan"]), "recovered")
        self.assertIn("OSError: unreadable", requests[-1]["messages"][-1]["content"])

    def test_router_uses_structured_output_and_rejects_unavailable_target(self):
        candidates = nodes.edges_from("Verify", exclude_kind="relocate")
        with model_script(calls(("Route", {"target": "Finish"}))) as requests:
            self.assertEqual(main.select_edge(candidates, main.REPAIR_INTENT, {}, {}, "Verify").target, "Finish")
        schema = requests[0]["tools"][0]["function"]["parameters"]
        self.assertEqual(schema["properties"]["target"]["enum"], ["Edit", "Finish"])
        for reply in [calls(("Route", {"target": "Explore"})), answer("unstructured")]:
            with model_script(reply, reply):
                self.assertIsNone(main.select_edge(candidates, main.REPAIR_INTENT, {}, {}, "Verify"))

    def test_router_retries_invalid_structure_once(self):
        with model_script(calls(("Route", {"target": "invalid"})),
                          calls(("Route", {"target": "Edit"}))) as requests:
            edge = main.select_edge(nodes.edges_from("Verify"), main.REPAIR_INTENT, {}, {}, "Verify")
        self.assertEqual(edge.target, "Edit")
        self.assertEqual(len(requests), 2)

    def test_evidence_guards_budgets_and_relocation_eligibility(self):
        state = {"issue": "fix", "repo_root": self.temp.name, "ledger": [],
                 "repairs": 0, "relocates": 0, "answer": "", "next": ""}
        with patch.object(main, "work", return_value="verified"), patch.object(main, "select_edge", return_value=None) as router:
            self.assertEqual(main.verify_node(state)["next"], "Edit")
            router.assert_not_called()
            nodes.log_call("replace_in_file")
            self.assertEqual(main.verify_node(state)["next"], "Edit")
            router.assert_not_called()
            nodes.log_call("run_snippet")
            self.assertEqual(main.verify_node(state)["next"], "Edit")
            self.assertEqual([e.kind for e in router.call_args.args[0]], ["repair", "finish"])
            relocated = main.verify_node({**state, "repairs": 1})
            self.assertEqual(relocated["relocates"], 1)
            self.assertEqual([e.kind for e in router.call_args.args[0]], ["repair", "relocate", "finish"])
            main.verify_node({**state, "repairs": 2, "relocates": 1})
            self.assertEqual([e.kind for e in router.call_args.args[0]], ["repair", "finish"])
            router.reset_mock()
            self.assertEqual(main.verify_node({**state, "repairs": 3})["next"], "Finish")
            nodes.tool_log[:] = [{"name": "read_file", "ok": True}] * 60
            self.assertEqual(main.verify_node(state)["next"], "Finish")
            router.assert_not_called()

    def test_workflow_stops_after_three_repair_opportunities(self):
        replies = [answer("plan"), answer("explore")] + [answer("edit"), answer("verify")] * 4
        with model_script(*replies) as requests:
            result = main.solve("fix", self.temp.name)
        self.assertEqual(result.count("### Edit\n"), 4)
        self.assertEqual(result.count("### Verify\nverify"), 4)
        self.assertEqual(len(requests), 10)

    def test_full_workflow_edits_runs_verifies_and_reports_usage(self):
        repro = "from calc import add; assert add(2, 3) == 5"
        replies = [
            answer("plan"),
            calls(("run_snippet", {"code": repro})), answer("located"),
            calls(("replace_in_file", {"path": "calc.py", "old": "return a - b", "new": "return a + b"})),
            calls(("run_snippet", {"code": repro})), answer("edited"),
            calls(("run_command", {"command": f"python -c '{repro}'"})), answer("verified"),
            calls(("Route", {"target": "Finish"})),
        ]
        trace = Path(self.temp.name, "trace.jsonl")
        with patch.dict(os.environ, {"CODEAGENT_TRACE": str(trace)}), model_script(*replies) as requests:
            result = orchestrator.solve("Fix add", self.temp.name)
        self.assertEqual(result.answer, "### Plan\nplan\n\n### Explore\nlocated\n\n### Edit\nedited\n\n### Verify\nverified")
        self.assertIn("return a + b", Path(self.temp.name, "calc.py").read_text())
        self.assertFalse(Path(self.temp.name, nodes.SCRATCH).exists())
        self.assertEqual(result.llm_calls, len(replies))
        self.assertEqual(result.prompt_tokens, 10 * len(replies))
        self.assertEqual(result.completion_tokens, 2 * len(replies))
        self.assertEqual(result.cached_tokens, 3 * len(replies))
        self.assertEqual(len(trace.read_text().splitlines()), len(replies))
        self.assertIn("Fix add", requests[0]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
