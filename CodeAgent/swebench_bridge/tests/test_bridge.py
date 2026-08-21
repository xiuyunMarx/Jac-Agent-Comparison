"""What the bridge has to get right for a number to mean anything.

Not a test of the agents -- they have their own suites -- and not of the
container mechanics, which need images. These cover the parts that decide what
gets reported: which paths count as a change, what the preparation step derives
from a dataset row, and how N graded runs are partitioned and rendered.

The comparison is tested at two, three *and* four sides on purpose. The previous
version of this directory was structurally two-sided, and every place that broke
when a third implementation arrived was a place that had only ever been exercised
with two.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

BRIDGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BRIDGE))

import frameworks  # noqa: E402
import grade  # noqa: E402
import report  # noqa: E402
import workspace  # noqa: E402


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------


def test_every_framework_points_at_a_real_agent():
    for name in frameworks.NAMES:
        fw = frameworks.get(name)
        assert fw.home.is_dir(), f"{name}: no agent directory at {fw.home}"
        assert (fw.home / fw.marker).exists(), f"{name}: no {fw.marker}"
        assert fw.entry.exists(), f"{name}: no shim at {fw.entry}"


def test_the_two_python_agents_share_one_shim():
    # Not an accident worth preserving by hand: the shim resolves its agent from
    # $CODEAGENT_HOME, so a second copy could drift and change what a run
    # records without changing either agent.
    assert (frameworks.get("langgraph").entry
            == frameworks.get("openai").entry)
    assert frameworks.get("byllm").entry != frameworks.get("openai").entry


def test_order_covers_the_registry():
    # A framework missing from ORDER would silently drop out of the default
    # comparison while still being runnable on its own.
    assert set(frameworks.ORDER) == set(frameworks.NAMES)


def test_ordered_is_stable_and_keeps_unknowns_last():
    assert frameworks.ordered(["openai", "byllm"]) == ["byllm", "openai"]
    assert frameworks.ordered(["zzz", "byllm"]) == ["byllm", "zzz"]


def test_unknown_framework_is_refused():
    with pytest.raises(SystemExit):
        frameworks.get("nope")


# --------------------------------------------------------------------------
# What counts as a change
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", [
    "build/lib/x.py", "dist/x.whl", "src/__pycache__/m.pyc", "x.egg-info/PKG-INFO",
    "sklearn/tree/_tree.so", "a/.pytest_cache/v/cache", "note.orig", "run.log",
    ".coverage",
])
def test_noise_is_excluded(path):
    assert workspace.is_noise(path)


@pytest.mark.parametrize("path", [
    "django/utils/text.py", "src/_pytest/mark/evaluate.py",
    "tests/test_text.py", "docs/build.rst", "lib/matplotlib/offsetbox.py",
])
def test_real_source_is_not_noise(path):
    # `docs/build.rst` is the interesting one: "build" is only noise as the
    # first component, not anywhere in the path.
    assert not workspace.is_noise(path)


def git_repo(tmp_path: Path) -> Path:
    run = lambda *a: subprocess.run(a, cwd=tmp_path, check=True,
                                    capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@t")
    run("git", "config", "user.name", "t")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def f():\n    return 1\n")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "base")
    return tmp_path


def test_extract_patch_reports_only_what_the_run_changed(tmp_path):
    ws = git_repo(tmp_path)
    # The image ships a dirty tree: this untracked artifact is present *before*
    # the agent starts, and must not reach the prediction.
    (ws / "pkg" / "built.so").write_bytes(b"\x00")
    (ws / "stale.txt").write_text("shipped dirty\n")
    baseline = workspace.porcelain(ws)

    (ws / "pkg" / "mod.py").write_text("def f():\n    return 2\n")
    (ws / "pkg" / "__pycache__").mkdir()
    (ws / "pkg" / "__pycache__" / "mod.pyc").write_bytes(b"\x00")

    patch = workspace.extract_patch(ws, baseline)
    assert "pkg/mod.py" in patch
    assert "return 2" in patch
    assert "__pycache__" not in patch
    assert "built.so" not in patch
    assert "stale.txt" not in patch


def test_extract_patch_is_empty_when_nothing_changed(tmp_path):
    ws = git_repo(tmp_path)
    assert workspace.extract_patch(ws, workspace.porcelain(ws)) == ""


def test_extract_patch_refuses_a_runaway_write(tmp_path, monkeypatch):
    ws = git_repo(tmp_path)
    baseline = workspace.porcelain(ws)
    monkeypatch.setattr(workspace, "MAX_PATCH_BYTES", 200)
    (ws / "pkg" / "mod.py").write_text("x = 1\n" * 500)
    with pytest.raises(Exception) as excinfo:
        workspace.extract_patch(ws, baseline)
    assert "over the" in str(excinfo.value)


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------

EVAL_SCRIPT = """\
#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
git status
git -c core.fileMode=false diff abc123
python -m pip install -e .
locale-gen en_US.UTF-8
git checkout abc123 tests/test_it.py
git apply -v - <<'EOF'
EOF
"""


def test_install_commands_stop_at_the_test_setup():
    cmds = workspace.install_commands(EVAL_SCRIPT)
    assert cmds == ["python -m pip install -e .", "locale-gen en_US.UTF-8"]
    # `git -c ... diff` looks like the start of the test setup and is not.
    assert not any("diff" in c for c in cmds)


def test_auto_prepare_skips_an_already_editable_install():
    # The bind mount does not break an editable install, and replaying it costs
    # minutes on the compiled repos.
    cmds = workspace.preparation_script({"eval_script": EVAL_SCRIPT}, "auto")
    assert cmds == ["locale-gen en_US.UTF-8"]


def test_auto_prepare_repairs_a_non_editable_install():
    script = EVAL_SCRIPT.replace("pip install -e .", "pip install .")
    cmds = workspace.preparation_script({"eval_script": script}, "auto")
    assert "python -m pip install -e ." in cmds


def test_prepare_modes():
    inst = {"eval_script": EVAL_SCRIPT}
    assert workspace.preparation_script(inst, "never") == []
    assert len(workspace.preparation_script(inst, "always")) == 2


@pytest.mark.parametrize("cmd,want", [
    ("pip install .", "pip install -e ."),
    ("python -m pip install .[test]", "python -m pip install -e .[test]"),
    ("pip install -e .", None),
    ("pip install --editable .", None),
    ("locale-gen en_US.UTF-8", None),
])
def test_as_editable(cmd, want):
    assert workspace.as_editable(cmd) == want


# --------------------------------------------------------------------------
# The partition
# --------------------------------------------------------------------------


def side(label: str, resolved: set[str], graded: set[str], **extra) -> dict:
    runs = {i: {"model": "gpt-5", "framework": label, "steps": 5,
                "llm_calls": 10, "tool_call_count": 8, "total_sec": 100,
                "prompt_tokens": 1000, "completion_tokens": 100}
            for i in graded}
    base = {
        "label": label, "report_path": Path("/x"), "resolved": resolved,
        "submitted": len(graded), "unresolved": len(graded - resolved),
        "empty": 0, "errored": 0, "empty_ids": set(), "error_ids": set(),
        "runs": runs, "won": [], "lost": [], "tokens": 1100 * len(graded),
        "cached_tokens": 0,
        "tokens_per_instance": 1100.0, "tokens_per_resolved": 2200.0,
        "llm_calls": 10.0, "steps": 5.0, "tool_calls": 8.0, "wall": 100.0,
        "model": "gpt-5", "model_name": f"{label}-codeagent/gpt-5",
        "framework": label,
    }
    base.update(extra)
    return base


ALL = {"a", "b", "c", "d", "e"}


def test_partition_files_every_instance_exactly_once():
    sides = [side("x", {"a", "b"}, ALL), side("y", {"b", "c"}, ALL),
             side("z", {"c"}, ALL)]
    graded, groups = report.partition(sides)
    assert graded == ALL
    assert sum(len(v) for v in groups.values()) == len(ALL)
    assert groups[(0,)] == {"a"}          # only x
    assert groups[(0, 1)] == {"b"}        # x and y
    assert groups[(1, 2)] == {"c"}        # y and z
    assert groups[()] == {"d", "e"}       # none


def test_group_order_is_complete_and_widest_first():
    for n in (2, 3, 4):
        keys = report.group_order(n)
        assert keys[0] == tuple(range(n))
        assert keys[-1] == ()
        assert len(keys) == 2 ** n
        assert len(set(keys)) == len(keys)
        sizes = [len(k) for k in keys]
        assert sizes == sorted(sizes, reverse=True)


def test_two_side_wording_stays_the_familiar_one():
    sides = [side("x", set(), set()), side("y", set(), set())]
    assert report.group_name((0, 1), sides) == "resolved by both"
    assert report.group_name((), sides) == "resolved by neither"
    assert report.group_name((0,), sides) == "only x"


def test_three_side_wording():
    sides = [side(n, set(), set()) for n in ("x", "y", "z")]
    assert report.group_name((0, 1, 2), sides) == "resolved by all"
    assert report.group_name((), sides) == "resolved by none"
    assert report.group_name((0, 2), sides) == "x + z"
    assert report.group_name((1,), sides) == "only y"


# --------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------


def test_two_way_table_reports_the_disagreement():
    sides = [side("byllm", {"a", "b"}, ALL), side("langgraph", {"b", "c"}, ALL)]
    out = report.compare_table(sides)
    assert "A/B" in out
    assert "resolved by both         : 1" in out
    assert "only byllm               : 1" in out
    assert "only langgraph           : 1" in out
    assert "resolved by neither      : 2" in out
    assert "      a" in out  # the disagreement set is listed, not just counted


def test_three_way_table_names_every_subset():
    sides = [side("byllm", {"a", "b"}, ALL), side("langgraph", {"b", "c"}, ALL),
             side("openai", {"b", "c", "d"}, ALL)]
    out = report.compare_table(sides)
    assert "3-way" in out
    assert "resolved by all" in out
    assert "langgraph + openai" in out
    assert "only byllm" in out
    assert "only openai" in out
    assert "resolved by none" in out
    # Three value columns, one per side.
    header = [ln for ln in out.splitlines() if "3-way" in ln][0]
    assert header.count("byllm") == 1 and header.count("openai") == 1


def test_empty_groups_are_still_reported():
    # A zero next to "only openai" is a result; a missing row reads as unrun.
    sides = [side("byllm", {"a"}, ALL), side("langgraph", {"a"}, ALL),
             side("openai", {"a"}, ALL)]
    out = report.compare_table(sides)
    assert "only openai" in out
    assert "resolved by all" in out


def test_four_sides_render():
    sides = [side(n, {"a"}, ALL) for n in ("w", "x", "y", "z")]
    out = report.compare_table(sides)
    assert "4-way" in out
    assert "resolved by all" in out


def test_a_comparison_needs_two_runs():
    with pytest.raises(ValueError):
        report.compare_table([side("x", set(), ALL)])


def test_framework_falls_back_to_the_model_name():
    # Runs graded before run_agent recorded a framework only carry it here.
    assert report.framework_of({}, {"model_name_or_path": "byllm-codeagent/gpt-4o"}) \
        == "byllm"
    assert report.framework_of({"i": {"framework": "openai"}}, {}) == "openai"
    assert report.framework_of({}, {}) == ""


# --------------------------------------------------------------------------
# Grading bookkeeping
# --------------------------------------------------------------------------


def test_build_report_buckets_every_instance():
    predictions = {i: {} for i in ("ok", "bad", "none", "boom")}
    results = {
        "ok": {"ok": {"patch_exists": True, "resolved": True}},
        "bad": {"bad": {"patch_exists": True, "resolved": False}},
        "none": {"none": {"patch_exists": False, "resolved": False}},
        "boom": {"boom": {"patch_exists": True, "resolved": False,
                          "error": "container died"}},
    }
    rep = grade.build_report(predictions, results, "x/gpt-5")
    assert rep["resolved_ids"] == ["ok"]
    assert rep["unresolved_ids"] == ["bad"]
    assert rep["empty_patch_ids"] == ["none"]
    assert rep["error_ids"] == ["boom"]
    assert rep["submitted_instances"] == 4
    # An empty patch and an error were never measured, so neither completed.
    assert rep["completed_instances"] == 2


def test_verdicts_round_trip_so_grading_resumes(tmp_path):
    path = tmp_path / "eval_results.jsonl"
    grade.record_verdict(path, "a", {"a": {"resolved": True}})
    grade.record_verdict(path, "b", {"b": {"resolved": False}})
    done = grade.load_done(path)
    assert set(done) == {"a", "b"}
    assert done["a"]["a"]["resolved"] is True


def test_report_filename_matches_the_harness(tmp_path):
    got = grade.report_path_for(tmp_path / "predictions.jsonl",
                                "byllm-codeagent/gpt-5", "run-7")
    assert got.name == "byllm-codeagent__gpt-5.run-7.json"


def test_locate_report_finds_it(tmp_path):
    (tmp_path / "predictions.jsonl").write_text("")
    rep = tmp_path / "byllm-codeagent__gpt-5.run-7.json"
    rep.write_text(json.dumps({"resolved_ids": [], "submitted_instances": 0}))
    assert report.locate_report(tmp_path / "predictions.jsonl", "run-7") == rep


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------


def test_a_missing_provider_key_stops_the_run(monkeypatch):
    # Without this the run pulls images, unpacks containers and runs the
    # preparation step for every instance before dying at the first LLM call --
    # then writes empty-patch predictions that satisfy the resume check, so the
    # next attempt skips them. It cost a real run before it was made fatal.
    import run_agent

    for key in run_agent.PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)
    args = run_agent.parse_args(["--framework", "byllm", "--run-id", "t"])
    with pytest.raises(SystemExit) as excinfo:
        run_agent.preflight(args)
    assert "no provider API key" in str(excinfo.value)


def test_allow_no_key_is_the_escape_hatch(monkeypatch):
    import run_agent

    for key in run_agent.PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)
    args = run_agent.parse_args(
        ["--framework", "byllm", "--run-id", "t", "--allow-no-key"])
    run_agent.preflight(args)  # must not raise


def test_any_one_provider_key_is_enough(monkeypatch):
    import run_agent

    for key in run_agent.PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    args = run_agent.parse_args(["--framework", "byllm", "--run-id", "t"])
    run_agent.preflight(args)  # must not raise
