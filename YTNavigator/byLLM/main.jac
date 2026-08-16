"""Headless benchmark runner for the byLLM YT-Navigator counterpart.

Reads the shared questions JSONL, runs each question through the ChatAgent
walker (fresh conversation per question; questions sharing a scenario_id
continue one conversation in file order), and writes one result record per
question in the shared benchmark schema (YT-Navigator/benchmark/schemas.py),
so the output is directly comparable with the LangGraph implementation via
YT-Navigator/benchmark/evaluate.py.

Configuration (env):
    YTNAV_QUESTIONS  questions JSONL (default ../YT-Navigator/benchmark/questions.jsonl)
    YTNAV_OUTPUT     results JSONL   (default results_byllm.jsonl)
    YTNAV_CHANNEL    channel id      (default: the only channel in the database)
    POSTGRES_*       database connection (same names as YT-Navigator's .env)
    OPENAI_API_KEY   OpenAI access for both models (default gpt-4o-mini / gpt-4o)
    INSTANT_LLM / POWERFUL_LLM   model overrides (same names as the original)

Run:  jac run main.jac   (from this directory)
"""

import os;
import json;
import time;
import uuid;
import from datetime { datetime, timezone }
import from nodes { ChatAgent, AgentAnswer }
import from tools {
    register_token_tracking, reset_tool_log, get_tool_calls,
    set_active_channel, llm_call_count, llm_calls_since
}
import from retrieval { resolve_channel, channel_info }


def load_questions(path: str) -> list[dict] {
    questions = [];
    f = open(path, "r");
    for line in f.read().split("\n") {
        line = line.strip();
        if not line or line.startswith("#") {
            continue;
        }
        questions.append(json.loads(line));
    }
    f.close();
    return questions;
}


def conversation_text(history: list[dict]) -> str {
    # Keep the last 3 exchanges - a plain-text stand-in for the original's
    # 1000-token message trimming.
    parts = [];
    for turn in history[-3:] {
        parts.append("User: " + str(turn["user"]));
        parts.append("Assistant: " + str(turn["assistant"] or ""));
    }
    return "\n".join(parts);
}


def answer_to_dict(answer: AgentAnswer) -> dict {
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
                ]
            } for v in answer.videos
        ]
    };
}


with entry {
    register_token_tracking();

    questions_path = os.environ.get("YTNAV_QUESTIONS", "../YT-Navigator/benchmark/questions.jsonl");
    output_path = os.environ.get("YTNAV_OUTPUT", "results_byllm.jsonl");

    channel_id = str(resolve_channel(os.environ.get("YTNAV_CHANNEL", "")));
    set_active_channel(channel_id);
    info = str(channel_info(channel_id));

    questions = load_questions(questions_path);
    run_id = uuid.uuid4().hex[:8];
    print(f"Run {run_id}: {len(questions)} questions against channel '{channel_id}' -> {output_path}");

    histories: dict[str, list[dict]] = {};
    error_count = 0;
    out = open(output_path, "w");
    for (idx, question) in enumerate(questions) {
        qid = str(question.get("id") or f"q{idx + 1}");
        scenario = question.get("scenario_id");
        thread_key = str(scenario) if scenario else qid;
        if thread_key not in histories {
            histories[thread_key] = [];
        }
        history = histories[thread_key];
        message = str(question["question"]);

        reset_tool_log();
        calls_before = llm_call_count();
        status = "ok";
        error = None;
        agent = ChatAgent(
            message=message,
            channel_info=info,
            conversation=conversation_text(history)
        );
        started = time.perf_counter();
        try {
            root spawn agent;
        } except Exception as e {
            status = "error";
            error = str(e);
            error_count += 1;
        }
        latency = round(time.perf_counter() - started, 3);
        llm_calls = llm_calls_since(calls_before);
        prompt_tokens = 0;
        completion_tokens = 0;
        for call in llm_calls {
            prompt_tokens += int(call.get("prompt_tokens") or 0);
            completion_tokens += int(call.get("completion_tokens") or 0);
        }

        route = agent.route if agent.route else None;
        fallback_events = list(agent.fallback_events);
        parse_failed = any([ev.get("event") == "output_parse_fallback" for ev in fallback_events]);
        answer_text = None;
        answer_raw = None;
        answer_parsed = False;
        cited = [];
        try {
            answer = agent.answer;
            if answer is not None {
                answer_text = answer.placeholder;
                answer_raw = json.dumps(answer_to_dict(answer));
                answer_parsed = not parse_failed;
                cited = [v.id for v in answer.videos if v.id];
                if status == "ok" {
                    history.append({"user": message, "assistant": answer_text});
                }
            }
        } except Exception as e {
            # A malformed answer must cost one question, never the whole run.
            if status == "ok" {
                status = "error";
                error = f"result extraction failed: {e}";
                error_count += 1;
            }
        }

        record = {
            "run_id": run_id,
            "framework": "byllm",
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
            "tool_calls": get_tool_calls(),
            "llm_calls": llm_calls,
            "fallback_events": fallback_events,
            "latency_s": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "timestamp": datetime.now(timezone.utc).isoformat()
        };
        out.write(json.dumps(record) + "\n");
        out.flush();
        detail = f"route={route}" if status == "ok" else str(error);
        total = prompt_tokens + completion_tokens;
        print(f"  [{idx + 1}/{len(questions)}] {qid}: {status} ({latency}s, {total} tokens) {detail}");
    }
    out.close();
    print(f"Done: {len(questions) - error_count}/{len(questions)} completed, {error_count} errors. Results: {output_path}");
}
