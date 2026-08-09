"""Verdict logic: LIFT, DynaRetarget success, smoothness.

Trials are synthesized here, so these pin the criteria themselves rather
than any recorded solve. numpy only — no solve venv needed.
"""

import json

import numpy as np
import pytest

from studio.solve import evaluate

NQ = 43


def _traj(n, box_z, box_xy=(0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0), jitter=0.0):
    """(n, 43) qpos with the box at a prescribed height profile."""
    q = np.zeros((n, NQ))
    q[:, 3] = 1.0                      # base quat w
    q[:, 36] = box_xy[0]
    q[:, 37] = box_xy[1]
    q[:, 38] = box_z
    q[:, 39:43] = np.array(quat)
    if jitter:
        rng = np.random.default_rng(0)
        q[:, 7:36] = rng.normal(0, jitter, (n, 29))
    return q


def _write_task(tmp_path, ref, res=None, flags=()):
    task = tmp_path / "clip_00"
    (task / "0").mkdir(parents=True)
    np.savez(task / "0/trajectory_kinematic.npz", qpos=ref)
    if res is not None:
        np.savez(task / "0/trajectory_mjwp.npz", qpos=res)
    (task / "task_info.json").write_text(json.dumps({"quality_flags": list(flags)}))
    return task


def test_no_result_yet_is_none(tmp_path):
    task = _write_task(tmp_path, _traj(50, np.linspace(0.5, 1.0, 50)))
    assert evaluate.evaluate_task(task) is None


def test_a_clean_lift_passes_both_criteria(tmp_path):
    ref_z = np.linspace(0.5, 1.0, 50)
    # a perfect solve: 60 Hz result holding each 30 Hz reference frame twice
    res_z = ref_z[np.arange(100) // 2]
    r = evaluate.evaluate_task(
        _write_task(tmp_path, _traj(50, ref_z), _traj(100, res_z)))

    assert r["lifted"] is True
    assert r["succ_dr"] is True
    assert r["op_mean"] == pytest.approx(0.0, abs=1e-12)


def test_a_box_that_never_rises_is_not_a_lift(tmp_path):
    ref = _traj(50, np.linspace(0.5, 1.0, 50))
    res = _traj(100, np.full(100, 0.5))              # stayed put
    r = evaluate.evaluate_task(_write_task(tmp_path, ref, res))
    assert r["lifted"] is False


def test_partial_lift_below_seventy_percent_fails(tmp_path):
    ref = _traj(50, np.linspace(0.5, 1.0, 50))       # ref_lift = 0.5
    res = _traj(100, np.linspace(0.5, 0.8, 100))     # sim_lift = 0.3 < 0.35
    assert evaluate.evaluate_task(_write_task(tmp_path, ref, res))["lifted"] is False


def test_lift_height_reached_but_box_left_far_away_fails(tmp_path):
    ref = _traj(50, np.linspace(0.5, 1.0, 50))
    res = _traj(100, np.linspace(0.5, 1.0, 100), box_xy=(1.0, 0.0))
    r = evaluate.evaluate_task(_write_task(tmp_path, ref, res))
    assert r["op_final"] == pytest.approx(1.0)
    assert r["lifted"] is False


def test_a_level_carry_is_not_a_lift_but_can_still_pass_dr(tmp_path):
    """The documented gotcha: LIFT measures box height change, so a
    walk-and-carry reads 'no' while tracking is fine."""
    ref = _traj(50, np.full(50, 0.8))
    res = _traj(100, np.full(100, 0.8))
    r = evaluate.evaluate_task(_write_task(tmp_path, ref, res))
    assert r["lifted"] is False       # ref_lift == 0, below LIFT_MIN_REF
    assert r["succ_dr"] is True


def test_rotation_error_beyond_25_degrees_fails_dr(tmp_path):
    ref = _traj(50, np.linspace(0.5, 1.0, 50))
    half = np.deg2rad(40.0) / 2       # 40 deg about z
    res = _traj(100, np.linspace(0.5, 1.0, 100),
                quat=(np.cos(half), 0.0, 0.0, np.sin(half)))
    r = evaluate.evaluate_task(_write_task(tmp_path, ref, res))
    assert r["orot_mean"] == pytest.approx(np.deg2rad(40.0), abs=1e-6)
    assert r["succ_dr"] is False


def test_smoothness_is_a_ratio_against_the_reference(tmp_path):
    ref = _traj(50, np.linspace(0.5, 1.0, 50), jitter=0.01)
    res = _traj(100, np.linspace(0.5, 1.0, 100), jitter=0.01)
    r = evaluate.evaluate_task(_write_task(tmp_path, ref, res))
    assert r["smooth"] > 0.0


def test_a_result_shorter_than_the_reference_still_evaluates(tmp_path):
    """The solve stops horizon_steps before the reference ends."""
    ref = _traj(80, np.linspace(0.5, 1.0, 80))
    res = _traj(100, np.linspace(0.5, 0.95, 100))
    assert evaluate.evaluate_task(_write_task(tmp_path, ref, res)) is not None


# ------------------------------------------------------------ verdict --

def _metrics(lifted, succ_dr):
    return {"lifted": lifted, "succ_dr": succ_dr}


def test_verdict_prefers_lift_then_dr():
    assert evaluate.verdict([("a", _metrics(True, True))]) == "LIFT"
    assert evaluate.verdict([("a", _metrics(False, True))]) == "DR"
    assert evaluate.verdict([("a", _metrics(False, False))]) == "failed"


def test_verdict_of_an_unsolved_run_is_error():
    assert evaluate.verdict([("a", None)]) == "error"
    assert evaluate.verdict([]) == "error"


def test_verdict_takes_the_best_of_several_tasks():
    rows = [("a", _metrics(False, False)), ("b", _metrics(True, True))]
    assert evaluate.verdict(rows) == "LIFT"


def test_format_table_reports_the_tally():
    rows = [("a", _metrics(True, True) | {
        "op_mean": 0.01, "orot_mean": 0.02, "bp_mean": 0.03,
        "final_box_z": 1.0, "ref_final_box_z": 1.0, "smooth": 1.2,
        "flags": "-"})]
    out = evaluate.format_table(rows)
    assert "YES" in out
    assert "lifted: 1/1 run" in out


def test_format_row_marks_a_missing_result():
    assert "(no result)" in evaluate.format_row("clip_00", None)
