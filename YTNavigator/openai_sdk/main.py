"""Headless benchmark runner for the no-framework YT-Navigator counterpart.

A line-for-line port of byLLM's `main.jac`: reads the shared questions JSONL,
runs each question through the agent (fresh conversation per question;
questions sharing a scenario_id continue one conversation in file order), and
writes one result record per question in the shared benchmark schema
(YT-Navigator/benchmark/schemas.py), so the output is scored side by side
with the byLLM and LangGraph implementations by eval/score.py, unchanged.

Configuration (env, same names as the other two sides):
    YTNAV_QUESTIONS  questions JSONL (default ../YT-Navigator/benchmark/questions.jsonl)
    YTNAV_OUTPUT     results JSONL   (default results_openai_sdk.jsonl)
    YTNAV_CHANNEL    channel id      (default: the only channel in the database)
    POSTGRES_*       database connection (same names as YT-Navigator's .env)
    OPENAI_API_KEY   OpenAI access for both models (default gpt-4o-mini / gpt-4o)
    INSTANT_LLM / POWERFUL_LLM   model overrides (same names as the original)

Run:  python main.py   (from this directory)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import agent
import llm
import tools

FRAMEWORK = "openai_sdk"


def load_questions(path: str) -> list[dict[str, Any]]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            questions.append(json.loads(line))
    return questions


def answer_to_dict(answer: agent.AgentAnswer) -> dict[str, Any]:
    return {
        "placeholder": answer.placeholder,
        "videos": [
            {
                "id": v.id,
                "title": v.title,
                "description": v.description,
                "thumbnail_url": v.thumbnail_url,
                "timestamps": [
                    {"start": t.start, "end": t.end, "description": t.description} for t in v.timestamps
                ],
            }
            for v in answer.videos
        ],
    }


def main() -> None:
    questions_path = os.environ.get("YTNAV_QUESTIONS", "../YT-Navigator/benchmark/questions.jsonl")
    output_path = os.environ.get("YTNAV_OUTPUT", "results_openai_sdk.jsonl")

    channel_id = str(tools.retrieval.resolve_channel(os.environ.get("YTNAV_CHANNEL", "")))
    tools.set_active_channel(channel_id)
    info = str(tools.retrieval.channel_info(channel_id))

    questions = load_questions(questions_path)
    run_id = uuid.uuid4().hex[:8]
    print(f"Run {run_id}: {len(questions)} questions against channel '{channel_id}' -> {output_path}")

    histories: dict[str, list[dict[str, Any]]] = {}
    error_count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for idx, question in enumerate(questions):
            qid = str(question.get("id") or f"q{idx + 1}")
            scenario = question.get("scenario_id")
            thread_key = str(scenario) if scenario else qid
            history = histories.setdefault(thread_key, [])
            message = str(question["question"])

            tools.reset_tool_log()
            calls_before = llm.llm_call_count()
            status = "ok"
            error = None
            # Created here and passed in, so the route and fallback events
            # decided before an exception survive into the record -- byLLM's
            # walker keeps them the same way.
            result = agent.ChatResult()
            started = time.perf_counter()
            try:
                agent.chat(message, info, history, result=result)
            except Exception as e:  # noqa: BLE001 - one bad question, not the run
                status = "error"
                error = str(e)
                error_count += 1
            latency = round(time.perf_counter() - started, 3)

            llm_calls = llm.llm_calls_since(calls_before)
            prompt_tokens = sum(int(c.get("prompt_tokens") or 0) for c in llm_calls)
            completion_tokens = sum(int(c.get("completion_tokens") or 0) for c in llm_calls)

            route = result.route or None
            fallback_events = list(result.fallback_events)
            parse_failed = any(ev.get("event") == "output_parse_fallback" for ev in fallback_events)
            answer_text = None
            answer_raw = None
            answer_parsed = False
            cited: list[str] = []
            answer = result.answer
            if answer is not None:
                answer_text = answer.placeholder
                answer_raw = json.dumps(answer_to_dict(answer))
                answer_parsed = not parse_failed
                cited = [v.id for v in answer.videos if v.id]
                if status == "ok":
                    history.append({"user": message, "assistant": answer_text})

            record = {
                "run_id": run_id,
                "framework": FRAMEWORK,
                "question_id": qid,
                "scenario_id": scenario,
                "thread_id": f"bench-{run_id}-{thread_key}",
                "question": message,
                "status": status,
                "error": error,
                "route": route,
                "answer_text": answer_text,
                "answer_raw": answer_raw,
                "answer_parsed": answer_parsed,
                "cited_video_ids": cited,
                "tool_calls": tools.get_tool_calls(),
                "llm_calls": llm_calls,
                "fallback_events": fallback_events,
                "latency_s": latency,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            detail = f"route={route}" if status == "ok" else str(error)
            total = prompt_tokens + completion_tokens
            print(f"  [{idx + 1}/{len(questions)}] {qid}: {status} ({latency}s, {total} tokens) {detail}")

    print(
        f"Done: {len(questions) - error_count}/{len(questions)} completed, "
        f"{error_count} errors. Results: {output_path}"
    )


if __name__ == "__main__":
    main()
