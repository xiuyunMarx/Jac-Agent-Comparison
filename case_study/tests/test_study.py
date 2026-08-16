"""The two places this directory turns run data into a claim.

`classify` decides what a run is said to have done to an instance, and
`select_instances` decides which instances the next comparison is allowed to be
judged on. Both were previously hand-applied judgements with no code behind
them, which is exactly why they are the parts worth pinning.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))

import build_study  # noqa: E402
import select_instances as sel  # noqa: E402


def entry(resolved=False, f2p_ok=0, f2p_bad=0, p2p_ok=0, p2p_bad=0,
          applied=True, exists=True, error=""):
    def bucket(ok, bad):
        return {"success": [f"ok{i}" for i in range(ok)],
                "failure": [f"bad{i}" for i in range(bad)]}

    out = {
        "patch_exists": exists,
        "patch_successfully_applied": applied,
        "resolved": resolved,
        "tests_status": {
            "FAIL_TO_PASS": bucket(f2p_ok, f2p_bad),
            "PASS_TO_PASS": bucket(p2p_ok, p2p_bad),
            "FAIL_TO_FAIL": bucket(0, 0),
            "PASS_TO_FAIL": bucket(0, 0),
        },
    }
    if error:
        out["error"] = error
    return out


# --------------------------------------------------------------------------
# The status vocabulary
# --------------------------------------------------------------------------


def test_resolved():
    assert build_study.classify(entry(resolved=True, f2p_ok=1, p2p_ok=10),
                                True) == build_study.RESOLVED


def test_regression_needs_a_complete_fix_first():
    # Fixed every F2P and broke a P2P: the diagnosis was right, the scope was not.
    assert build_study.classify(entry(f2p_ok=2, p2p_ok=8, p2p_bad=5),
                                False) == build_study.REGRESSION


def test_a_wrong_patch_that_also_breaks_things_is_not_a_regression():
    # F2P still failing, so this is simply a wrong patch. Calling it a
    # regression would credit it with a fix it did not make.
    assert build_study.classify(entry(f2p_ok=0, f2p_bad=1, p2p_ok=18, p2p_bad=59),
                                False) == build_study.TESTS_FAIL


def test_a_collapsed_suite_is_not_a_regression():
    # 0 of 862 passing is not 862 regressions -- nothing ran.
    assert build_study.classify(entry(f2p_bad=1, p2p_ok=0, p2p_bad=862),
                                False) == build_study.SUITE_ERROR


def test_plain_failure():
    assert build_study.classify(entry(f2p_bad=1, p2p_ok=30),
                                False) == build_study.TESTS_FAIL


def test_unapplied_and_empty_and_errored():
    assert build_study.classify(entry(applied=False), False) == build_study.APPLY_FAIL
    assert build_study.classify(entry(exists=False), False) == build_study.EMPTY_PATCH
    assert build_study.classify(entry(error="boom"), False) == build_study.HARNESS_ERROR


def test_the_report_wins_when_there_is_no_log():
    # A grade that died before capturing a log still has to be reported as what
    # it was, not silently downgraded to a test failure.
    assert build_study.classify(None, False, errored=True) == build_study.HARNESS_ERROR
    assert build_study.classify(None, False, empty=True) == build_study.EMPTY_PATCH
    assert build_study.classify(None, True) == build_study.RESOLVED


def test_files_touched_reads_the_b_side():
    patch = ("diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n"
             "+++ b/pkg/mod.py\n@@ -1 +1 @@\n-x\n+y\n"
             "diff --git a/other.py b/other.py\n")
    assert build_study.files_touched(patch) == ["other.py", "pkg/mod.py"]
    assert build_study.files_touched("") == []


@pytest.mark.parametrize("raw,kind", [
    ("timed out after 1800s: jac run", "timeout"),
    ("`udocker create --name=x ...` exited 1", "infra(container)"),
    ("Error code: 400 - unanswered tool_call_id", "api-400"),
    ("", ""),
])
def test_error_kinds(raw, kind):
    assert build_study.error_kind(raw) == kind


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def row(iid, winner, identical="False", **errors):
    out = {"instance_id": iid, "repo": iid.split("__")[0],
           "resolved_by": winner, "identical_patches": identical,
           "byllm_run_error": "", "langgraph_run_error": ""}
    out.update(errors)
    return out


def test_identical_patches_are_not_evidence():
    r = row("sympy__sympy-24152", "langgraph", identical="True")
    assert "identical" in sel.excluded(r, ["byllm", "langgraph"])


def test_infrastructure_losses_are_not_evidence():
    r = row("django__django-12184", "byllm", byllm_run_error="",
            langgraph_run_error="infra(container)")
    assert "infrastructure" in sel.excluded(r, ["byllm", "langgraph"])


def test_a_timeout_is_evidence():
    # The agent spent its own budget and came back with nothing. That is a
    # result about the agent, not about the machine.
    r = row("django__django-13925", "byllm", byllm_run_error="timeout")
    assert sel.excluded(r, ["byllm", "langgraph"]) == ""


def test_draw_is_balanced_across_sides():
    pool = ([row(f"a__a-{i}", "byllm") for i in range(10)]
            + [row(f"b__b-{i}", "langgraph") for i in range(10)])
    got = sel.draw(pool, 8, set())
    assert len(got) == 8
    assert sum(1 for i in got if i.startswith("a__")) == 4
    assert sum(1 for i in got if i.startswith("b__")) == 4


def test_draw_caps_a_dominant_repo():
    pool = ([row(f"django__django-{i}", "byllm") for i in range(20)]
            + [row(f"sympy__sympy-{i}", "byllm") for i in range(20)])
    got = sel.draw(pool, 10, set(), max_repo_fraction=0.5)
    assert sum(1 for i in got if i.startswith("django__")) <= 5


def test_keep_pins_an_instance_past_the_cap():
    pool = ([row(f"django__django-{i}", "byllm") for i in range(20)]
            + [row(f"sympy__sympy-{i}", "byllm") for i in range(20)])
    got = sel.draw(pool, 4, {"django__django-19"}, max_repo_fraction=0.25)
    assert "django__django-19" in got


def test_asking_for_the_whole_pool_returns_the_whole_pool():
    # An even per-side quota cannot be met by an uneven pool: 12 wanted from a
    # 5/7 split gives each side 6, and the short side can only supply 5. Without
    # redistribution this returns 11 -- and "run every diverging instance" is
    # exactly the request that must not silently drop one.
    pool = ([row(f"a__a-{i}", "byllm") for i in range(5)]
            + [row(f"b__b-{i}", "langgraph") for i in range(7)])
    got = sel.draw(pool, 12, set(), max_repo_fraction=1.0)
    assert len(got) == 12
    assert set(got) == {r["instance_id"] for r in pool}


def test_asking_for_more_than_the_pool_is_capped_at_the_pool():
    pool = [row(f"a__a-{i}", "byllm") for i in range(4)]
    assert len(sel.draw(pool, 99, set(), max_repo_fraction=1.0)) == 4


def test_draw_is_deterministic():
    pool = [row(f"a__a-{i}", "byllm") for i in range(10)]
    assert sel.draw(pool, 5, set()) == sel.draw(pool, 5, set())


def test_frameworks_are_read_off_the_header():
    header = ["instance_id", "byllm_status", "openai_status", "gold_patch_bytes"]
    assert sel.frameworks_in(header) == ["byllm", "openai"]


# --------------------------------------------------------------------------
# The generated study, if it is present
# --------------------------------------------------------------------------

LITE = STUDY / "lite-01" / "divergence.csv"


@pytest.mark.skipif(not LITE.exists(), reason="lite-01 study not generated")
def test_the_generated_study_agrees_with_the_run_reports():
    import json

    rows = list(csv.DictReader(LITE.open(newline="", encoding="utf-8")))
    runs = STUDY.parent / "swebench_bridge" / "results"
    reports = {}
    for fw, d in (("byllm", "lite-01-byllm"), ("langgraph", "lite-01-langgraph")):
        path = next((runs / d).glob("*.lite-01-*.json"))
        reports[fw] = set(json.loads(path.read_text())["resolved_ids"])

    for r in rows:
        for fw in ("byllm", "langgraph"):
            resolved = r["instance_id"] in reports[fw]
            assert (r[f"{fw}_status"] == "RESOLVED") == resolved, r["instance_id"]
        # Diverging means exactly one side resolved it.
        assert r["resolved_by"] in ("byllm", "langgraph")


@pytest.mark.skipif(not LITE.exists(), reason="lite-01 study not generated")
def test_selection_excludes_exactly_the_non_evidence():
    rows = list(csv.DictReader(LITE.open(newline="", encoding="utf-8")))
    names = ["byllm", "langgraph"]
    dropped = {r["instance_id"] for r in rows if sel.excluded(r, names)}
    assert dropped == {
        "django__django-12184",
        "matplotlib__matplotlib-23314",
        "psf__requests-2317",
        "scikit-learn__scikit-learn-13439",
        "sympy__sympy-24152",
        "sympy__sympy-24213",
    }
