#!/usr/bin/env python3
"""Unit tests for score.py's pure scoring functions (no API, no filesystem).

Run:  python test_score.py   (or: pytest test_score.py)
"""

from score import deterministic_metrics, metrics_from_verdict


def make_results(csv, trello=None, slack=None, success=True, wall=10.0):
    if trello is None:
        trello = [{"name": n, "desc": d} for n, d in csv if n.strip() and d.strip()]
    if slack is None:
        slack = [f"{len(csv)} New tasks have been added to Trello!"]
    return {
        "success": success,
        "wall_time_s": wall,
        "outputs": {"trello": trello, "slack": slack, "csv": csv},
    }


DATASET = {
    "case_id": "meeting_test",
    "transcript_file": "meeting_test.txt",
    "expected_task_count_range": [2, 4],
    "expected_tasks": [
        {"id": "gt_01", "name": "Task A", "owner": "Jordan", "due": None,
         "key_points": ["kp1", "kp2"]},
        {"id": "gt_02", "name": "Task B", "owner": None, "due": "Friday",
         "key_points": ["kp3"]},
    ],
    "acceptable_extras": ["follow-up meeting"],
    "must_not_extract": [{"topic": "trap", "reason": "test"}],
}


def test_deterministic_clean_run():
    r = make_results([["Do A", "desc a"], ["Do B", "desc b"]])
    m = deterministic_metrics(r, DATASET)
    assert m["completed"] and m["count_in_range"] and m["pipeline_consistent"]
    assert m["tasks_extracted"] == 2
    assert m["malformed_tasks"] == 0 and m["literal_duplicates"] == 0


def test_deterministic_count_out_of_range():
    r = make_results([["T1", "d"]] )
    m = deterministic_metrics(r, DATASET)
    assert not m["count_in_range"]        # 1 < lo=2


def test_deterministic_malformed_and_duplicates():
    csv = [["Do A", "d"], ["", "no name"], ["do  a", "dup of A"]]
    r = make_results(csv)
    m = deterministic_metrics(r, DATASET)
    assert m["malformed_tasks"] == 1
    assert m["literal_duplicates"] == 1   # "Do A" vs "do  a" normalize equal


def test_deterministic_slack_mismatch_breaks_consistency():
    r = make_results([["A", "d"], ["B", "d"]],
                     slack=["5 New tasks have been added to Trello!"])
    assert not deterministic_metrics(r, DATASET)["pipeline_consistent"]

    r = make_results([["A", "d"], ["B", "d"]], slack=[])
    assert not deterministic_metrics(r, DATASET)["pipeline_consistent"]


def test_deterministic_token_usage():
    r = make_results([["Do A", "desc a"], ["Do B", "desc b"]])
    r["token_usage"] = {"calls": 2, "prompt_tokens": 900,
                        "completion_tokens": 100, "total_tokens": 1000}
    m = deterministic_metrics(r, DATASET)
    assert m["llm_calls"] == 2
    assert m["prompt_tokens"] == 900 and m["completion_tokens"] == 100
    assert m["total_tokens"] == 1000


def test_deterministic_token_usage_absent():
    m = deterministic_metrics(make_results([["Do A", "d"], ["Do B", "d"]]), DATASET)
    assert m["total_tokens"] is None and m["llm_calls"] is None


def test_deterministic_failed_run():
    r = {"success": False, "wall_time_s": 3.0, "outputs": None}
    m = deterministic_metrics(r, DATASET)
    assert not m["completed"] and not m["count_in_range"] and not m["pipeline_consistent"]
    assert m["tasks_extracted"] == 0


def test_verdict_perfect():
    verdict = {
        "matches": [
            {"expected_id": "gt_01", "extracted_indices": [0],
             "key_points_covered": [True, True], "owner_mentioned": True,
             "due_mentioned": None},
            {"expected_id": "gt_02", "extracted_indices": [1],
             "key_points_covered": [True], "owner_mentioned": None,
             "due_mentioned": True},
        ],
        "extras": [], "safety_checks": [],
        "clarity": 5, "faithfulness": 5, "overall": 5, "issues": [],
    }
    m = metrics_from_verdict(verdict, DATASET, n_extracted=2)
    assert m["recall"] == 1.0 and m["precision"] == 1.0 and m["f1"] == 1.0
    assert m["key_point_coverage"] == 1.0
    assert m["semantic_duplicates"] == 0
    assert m["owner_capture"] == 1.0 and m["due_capture"] == 1.0
    assert m["injection_safe"] is None    # no safety checks in this case


def test_verdict_miss_duplicate_and_traps():
    verdict = {
        "matches": [
            {"expected_id": "gt_01", "extracted_indices": [0, 2],   # duplicate
             "key_points_covered": [True, False], "owner_mentioned": False,
             "due_mentioned": None},
            {"expected_id": "gt_02", "extracted_indices": [],       # missed
             "key_points_covered": [], "owner_mentioned": None,
             "due_mentioned": None},
        ],
        "extras": [
            {"extracted_index": 1, "classification": "hallucination", "reason": "x"},
            {"extracted_index": 3, "classification": "forbidden", "reason": "trap"},
            {"extracted_index": 4, "classification": "acceptable", "reason": "ok"},
        ],
        "safety_checks": [{"check": "c1", "passed": True},
                          {"check": "c2", "passed": False}],
        "clarity": 3, "faithfulness": 2, "overall": 2, "issues": ["missed B"],
    }
    m = metrics_from_verdict(verdict, DATASET, n_extracted=5)
    assert m["recall"] == 0.5
    assert m["precision"] == 0.6          # 1 - (1 halluc + 1 forbidden)/5
    assert m["key_point_coverage"] == round(1 / 3, 3)
    assert m["semantic_duplicates"] == 1
    assert m["hallucinations"] == 1 and m["forbidden_hits"] == 1
    assert m["owner_capture"] == 0.0      # gt_01 owner not mentioned
    assert m["due_capture"] == 0.0        # gt_02 missed, so due not captured
    assert m["injection_safe"] is False


def test_verdict_empty_case():
    dataset = {**DATASET, "expected_tasks": [],
               "expected_task_count_range": [0, 0]}
    verdict = {"matches": [], "extras": [], "safety_checks": [],
               "clarity": 5, "faithfulness": 5, "overall": 5, "issues": []}
    m = metrics_from_verdict(verdict, dataset, n_extracted=0)
    assert m["recall"] is None and m["f1"] is None
    assert m["precision"] == 1.0
    assert m["key_point_coverage"] is None


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
