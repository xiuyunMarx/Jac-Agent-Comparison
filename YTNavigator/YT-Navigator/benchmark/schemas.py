"""Data contracts for the YT-Navigator agent benchmark.

These are deliberately framework-agnostic and dependency-free (stdlib only):
any agent implementation under comparison (LangGraph, Jac, ...) must consume
the same question format and emit the same result format so that
``benchmark/evaluate.py`` can score them side by side.

Question record (one JSON object per line in a ``.jsonl`` file):

    {
      "id": "q1",                       # optional, auto-assigned "q<N>" if missing
      "question": "...",                # required
      "expected_route": "Yes",          # optional: "Yes" (tool call) | "No" (direct) | "Not relevant" (refusal)
      "expected_video_ids": ["abc123"], # optional: video ids a correct answer should cite
      "reference_answer": "...",        # optional: gold answer for LLM-judge scoring
      "scenario_id": "s1"               # optional: questions sharing a scenario_id run in the
                                        # same conversation thread, in file order
    }

Result record (one JSON object per line, produced by the runner):
see ``make_result`` below for the canonical field list.
"""

import json
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

VALID_ROUTES = ("Yes", "No", "Not relevant")


@dataclass
class BenchmarkQuestion:
    """A single benchmark question with optional grading references."""

    id: str
    question: str
    expected_route: Optional[str] = None
    expected_video_ids: List[str] = field(default_factory=list)
    reference_answer: Optional[str] = None
    scenario_id: Optional[str] = None


def load_questions(path: str) -> List[BenchmarkQuestion]:
    """Load and validate a questions JSONL file.

    Args:
        path: Path to the questions file.

    Returns:
        List of validated BenchmarkQuestion objects, in file order.

    Raises:
        ValueError: On malformed JSON, missing question text, or invalid route.
    """
    questions = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {e}") from e

            text = (raw.get("question") or "").strip()
            if not text:
                raise ValueError(f"{path}:{line_number}: missing 'question' field")

            expected_route = raw.get("expected_route")
            if expected_route is not None and expected_route not in VALID_ROUTES:
                raise ValueError(
                    f"{path}:{line_number}: expected_route must be one of {VALID_ROUTES}, got {expected_route!r}"
                )

            questions.append(
                BenchmarkQuestion(
                    id=str(raw.get("id") or f"q{len(questions) + 1}"),
                    question=text,
                    expected_route=expected_route,
                    expected_video_ids=[str(v) for v in raw.get("expected_video_ids") or []],
                    reference_answer=raw.get("reference_answer"),
                    scenario_id=raw.get("scenario_id"),
                )
            )

    ids = [q.id for q in questions]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"{path}: duplicate question ids: {sorted(duplicates)}")
    return questions


def make_result(
    *,
    run_id: str,
    framework: str,
    question: BenchmarkQuestion,
    thread_id: str,
    status: str,
    timestamp: str,
    latency_s: float,
    error: Optional[str] = None,
    route: Optional[str] = None,
    answer_text: Optional[str] = None,
    answer_raw: Optional[str] = None,
    answer_parsed: bool = False,
    cited_video_ids: Optional[List[str]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    llm_calls: Optional[List[Dict[str, Any]]] = None,
    fallback_events: Optional[List[Dict[str, Any]]] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> Dict[str, Any]:
    """Build a canonical result record.

    Every implementation under comparison must emit records with exactly these
    fields (unknown extra fields are tolerated by the evaluator but ignored).

    Args:
        run_id: Identifier shared by all records of one run.
        framework: Label of the implementation, e.g. "langgraph" or "jac".
        question: The question this result answers.
        thread_id: Conversation thread the question ran in.
        status: "ok" or "error".
        timestamp: ISO-8601 UTC time the question finished.
        latency_s: End-to-end wall-clock seconds for the invocation.
        error: Error message when status is "error".
        route: Router decision ("Yes" | "No" | "Not relevant") or None.
        answer_text: Human-readable answer text (markup stripped).
        answer_raw: Raw final message content as produced by the agent.
        answer_parsed: Whether the final answer conformed to the output schema.
        cited_video_ids: Video ids cited in the structured answer.
        tool_calls: [{"name": ..., "args": ...}] tool invocations made.
        llm_calls: [{"model", "latency_s", "prompt_tokens", "completion_tokens"}] per LLM call.
        fallback_events: Silent error-recovery events recorded during the run.
        prompt_tokens: Total prompt tokens across all LLM calls.
        completion_tokens: Total completion tokens across all LLM calls.

    Returns:
        A JSON-serializable dict, one line of the results JSONL file.
    """
    return {
        "run_id": run_id,
        "framework": framework,
        "question_id": question.id,
        "scenario_id": question.scenario_id,
        "thread_id": thread_id,
        "question": question.question,
        "status": status,
        "error": error,
        "route": route,
        "answer_text": answer_text,
        "answer_raw": answer_raw,
        "answer_parsed": answer_parsed,
        "cited_video_ids": cited_video_ids or [],
        "tool_calls": tool_calls or [],
        "llm_calls": llm_calls or [],
        "fallback_events": fallback_events or [],
        "latency_s": latency_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "timestamp": timestamp,
    }


def load_results(path: str) -> List[Dict[str, Any]]:
    """Load a results JSONL file produced by a benchmark runner.

    Args:
        path: Path to the results file.

    Returns:
        List of result record dicts, in file order.
    """
    results = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {e}") from e
    return results
