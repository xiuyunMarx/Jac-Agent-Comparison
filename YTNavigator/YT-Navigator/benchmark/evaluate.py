#!/usr/bin/env python3
"""Score and compare benchmark result files.

Takes one or more results JSONL files (each produced by an agent
implementation via the shared contract in schemas.py) and reports per-run
metrics side by side:

- completion / error rate and silent-fallback rate
- routing accuracy (vs expected_route in the question set)
- retrieval hit rate and recall (cited vs expected_video_ids)
- structured-output conformance (answer_parsed)
- latency mean / p50 / p95, token usage, LLM- and tool-call counts
- optional LLM-as-judge answer quality (needs OPENAI_API_KEY and reference_answer)

This script is intentionally Django-free so it can score any implementation.

Usage:
    python benchmark/evaluate.py results/langgraph.jsonl results/jac.jsonl \
        --questions benchmark/questions.jsonl [--judge] [--report report.json]
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.schemas import (  # noqa: E402
    load_questions,
    load_results,
)

JUDGE_PROMPT = """You are grading an AI assistant that answers questions about a YouTube channel's content.

Question: {question}

Reference answer (ground truth): {reference}

Assistant's answer: {answer}

Score the assistant's answer from 1 to 5 for factual agreement with the reference:
5 = fully consistent and complete, 4 = consistent with minor omissions,
3 = partially correct, 2 = mostly incorrect or evasive, 1 = wrong or hallucinated.

Reply with ONLY a JSON object: {{"score": <1-5>, "reasoning": "<one sentence>"}}"""


def percentile(values, fraction):
    """Return the interpolated percentile of a list of numbers."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def score_run(results, questions_by_id):
    """Compute all metrics for one results file.

    Args:
        results: List of result records.
        questions_by_id: Mapping of question id -> BenchmarkQuestion (may be empty).

    Returns:
        Dict of metric name -> value (None when not applicable).
    """
    completed = [r for r in results if r.get("status") == "ok"]
    n = len(results)
    metrics = {
        "questions": n,
        "completed": len(completed),
        "error_rate": round(1 - len(completed) / n, 3) if n else None,
        "fallback_rate": round(sum(1 for r in results if r.get("fallback_events")) / n, 3) if n else None,
    }

    # Routing accuracy
    routed = [
        (r, questions_by_id[r["question_id"]].expected_route)
        for r in completed
        if r.get("question_id") in questions_by_id and questions_by_id[r["question_id"]].expected_route
    ]
    metrics["routing_scored"] = len(routed)
    metrics["routing_accuracy"] = (
        round(sum(1 for r, expected in routed if r.get("route") == expected) / len(routed), 3) if routed else None
    )

    # Retrieval quality: did the answer cite the expected videos?
    hits, recalls = [], []
    for r in completed:
        question = questions_by_id.get(r.get("question_id"))
        if question is None or not question.expected_video_ids:
            continue
        expected = set(question.expected_video_ids)
        cited = set(r.get("cited_video_ids") or [])
        hits.append(bool(expected & cited))
        recalls.append(len(expected & cited) / len(expected))
    metrics["retrieval_scored"] = len(hits)
    metrics["retrieval_hit_rate"] = round(sum(hits) / len(hits), 3) if hits else None
    metrics["retrieval_recall"] = round(sum(recalls) / len(recalls), 3) if recalls else None

    # Structured output conformance
    metrics["answer_parse_rate"] = (
        round(sum(1 for r in completed if r.get("answer_parsed")) / len(completed), 3) if completed else None
    )

    # Latency
    latencies = [r["latency_s"] for r in completed if isinstance(r.get("latency_s"), (int, float))]
    metrics["latency_mean_s"] = round(statistics.mean(latencies), 2) if latencies else None
    metrics["latency_p50_s"] = round(percentile(latencies, 0.5), 2) if latencies else None
    metrics["latency_p95_s"] = round(percentile(latencies, 0.95), 2) if latencies else None

    # Tokens and call counts
    tokens = [r.get("total_tokens") or 0 for r in completed]
    metrics["tokens_total"] = sum(tokens) if tokens else None
    metrics["tokens_mean"] = round(statistics.mean(tokens), 1) if tokens else None
    metrics["llm_calls_mean"] = (
        round(statistics.mean([len(r.get("llm_calls") or []) for r in completed]), 2) if completed else None
    )
    metrics["tool_calls_mean"] = (
        round(statistics.mean([len(r.get("tool_calls") or []) for r in completed]), 2) if completed else None
    )
    return metrics


def judge_run(results, questions_by_id, model_name):
    """Score answer quality with an LLM judge (1-5 vs reference answers).

    Args:
        results: List of result records.
        questions_by_id: Mapping of question id -> BenchmarkQuestion.
        model_name: litellm model name to judge with (default gpt-4o-mini;
            plain names are OpenAI, "provider/model" selects another provider).

    Returns:
        (mean score or None, per-question {question_id: score}).
    """
    try:
        import litellm
    except ImportError:
        print("warning: litellm not installed, skipping judge (pip install litellm)", file=sys.stderr)
        return None, {}
    if "/" not in model_name and not os.getenv("OPENAI_API_KEY"):
        print("warning: OPENAI_API_KEY not set, skipping judge", file=sys.stderr)
        return None, {}

    scores = {}
    for r in results:
        question = questions_by_id.get(r.get("question_id"))
        if r.get("status") != "ok" or question is None or not question.reference_answer:
            continue
        prompt = JUDGE_PROMPT.format(
            question=question.question,
            reference=question.reference_answer,
            answer=r.get("answer_text") or r.get("answer_raw") or "(no answer)",
        )
        try:
            reply = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            ).choices[0].message.content
            payload = json.loads(reply[reply.index("{") : reply.rindex("}") + 1])
            score = int(payload["score"])
            if 1 <= score <= 5:
                scores[r["question_id"]] = score
        except Exception as e:
            print(f"warning: judge failed on {r.get('question_id')}: {e}", file=sys.stderr)

    mean = round(statistics.mean(scores.values()), 2) if scores else None
    return mean, scores


METRIC_ROWS = [
    ("questions", "Questions"),
    ("completed", "Completed"),
    ("error_rate", "Error rate"),
    ("fallback_rate", "Fallback rate"),
    ("routing_accuracy", "Routing accuracy"),
    ("retrieval_hit_rate", "Retrieval hit rate"),
    ("retrieval_recall", "Retrieval recall"),
    ("answer_parse_rate", "Answer parse rate"),
    ("judge_score_mean", "Judge score (1-5)"),
    ("latency_mean_s", "Latency mean (s)"),
    ("latency_p50_s", "Latency p50 (s)"),
    ("latency_p95_s", "Latency p95 (s)"),
    ("tokens_mean", "Tokens / question"),
    ("tokens_total", "Tokens total"),
    ("llm_calls_mean", "LLM calls / question"),
    ("tool_calls_mean", "Tool calls / question"),
]


def print_table(runs):
    """Print runs side by side, one metric per row."""
    labels = [label for label, _ in runs]
    name_width = max(len(name) for _, name in METRIC_ROWS) + 2
    col_width = max(12, max(len(label) for label in labels) + 2)

    header = " " * name_width + "".join(label.rjust(col_width) for label in labels)
    print(header)
    print("-" * len(header))
    for key, name in METRIC_ROWS:
        if all(metrics.get(key) is None for _, metrics in runs):
            continue
        cells = "".join(
            (str(metrics.get(key)) if metrics.get(key) is not None else "-").rjust(col_width)
            for _, metrics in runs
        )
        print(name.ljust(name_width) + cells)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="One or more results JSONL files to score and compare")
    parser.add_argument("--questions", help="Questions JSONL with expected_route / expected_video_ids / references")
    parser.add_argument("--judge", action="store_true", help="Also run LLM-as-judge scoring (needs OPENAI_API_KEY)")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="litellm model name used as judge")
    parser.add_argument("--report", help="Write the full metrics (and per-question judge scores) to this JSON file")
    args = parser.parse_args()

    questions_by_id = {}
    if args.questions:
        questions_by_id = {q.id: q for q in load_questions(args.questions)}
    else:
        print("note: no --questions file given; routing/retrieval/judge metrics are skipped", file=sys.stderr)

    runs = []
    report = {}
    for path in args.results:
        results = load_results(path)
        label = (results[0].get("framework") if results else None) or Path(path).stem
        metrics = score_run(results, questions_by_id)
        judge_scores = {}
        if args.judge:
            metrics["judge_score_mean"], judge_scores = judge_run(results, questions_by_id, args.judge_model)
        runs.append((label, metrics))
        report[label] = {"file": path, "metrics": metrics, "judge_scores": judge_scores}

    print_table(runs)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report written to {args.report}")


if __name__ == "__main__":
    main()
