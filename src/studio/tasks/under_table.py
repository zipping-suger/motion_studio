"""Under-table pick: collision-free augmentation against an inferred table.

The clip is an under-table pick (robot ducks and reaches under a surface
that isn't there); `recon.table` places a randomized legged table over the
duck from head-trajectory FK, and `solve.mppi_loop` re-solves the motion so
the robot tracks the reference while actually avoiding the now-solid table.

Verification (numpy + mujoco, studio venv): SDF penetration of the executed
trajectory, pelvis drift vs the reference, and the pick-hand
task-preservation check — a solve can go "collision-free" by not reaching
under the table at all, and only the pick error catches that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import solve
from ..config import PROCESSED_ROOT
from ..recon import table
from . import overrides
from .base import Task
from .under_table_params import (SCENE_DEFAULTS, SOLVE_DEFAULTS,
                                 SOLVE_INT_KEYS, VERIFY_DEFAULTS, coerce)

SUBTREE = PROCESSED_ROOT / "humanoid"   # robot-only embodiment
RESULT_NPZ = "trajectory_aug.npz"


def _merge(defaults: dict, *layers) -> dict:
    """Defaults + override layers, unknown keys rejected, values coerced to
    the default's type (CLI --set and YAML both funnel through here)."""
    out = dict(defaults)
    for layer in layers:
        for key, val in (layer or {}).items():
            if key not in defaults:
                raise SystemExit(f"unknown under_table parameter: {key}")
            out[key] = coerce(val, defaults[key])
    return out


# ------------------------------------------------------------------ recon --

def _reconstruct(npz, out_root, task, options):
    options = options or {}
    params = _merge(SCENE_DEFAULTS, overrides("under_table", "scene"),
                    options.get("scene_params"))
    task_dir, _spec, line = table.reconstruct(npz, out_root, task, params)
    return task_dir, line


# ------------------------------------------------------------------ solve --

def _solve_command(cfg, task, dataset_dir, params):
    solve.require(cfg)
    task_dir = Path(dataset_dir) / SUBTREE / task
    args = [str(cfg.solve_python), "-m", "studio.solve.mppi_loop",
            "--task-dir", str(task_dir)]
    for key, value in (params or {}).items():
        args += ["--param", f"{key}={value}"]
    return args


# --------------------------------------------------------------- evaluate --

def _evaluate_task(task_dir: Path) -> dict | None:
    res_path = task_dir / "0" / RESULT_NPZ
    if not res_path.exists():
        return None
    info = json.loads((task_dir / "task_info.json").read_text())
    res = np.load(res_path)
    aug, ref = res["aug_qpos"], res["ref_qpos_interp"]
    if not len(aug):
        return None

    model = table.robot_model()
    points, radii = table.trajectory_collision_points(aug, model)
    clearance = table.clearance_profile_prims(points, radii,
                                              info["primitives"])
    min_clear = float(clearance.min())
    pos_err, rot_err = table.root_pose_errors(aug, ref)
    pick_err = table.pick_hand_error(aug, ref, model)

    thr = _merge(VERIFY_DEFAULTS, overrides("under_table", "verify"))
    penetration = max(0.0, -min_clear)
    return {
        "penetration": penetration,
        "pen_frames": int((clearance < 0).sum()),
        "frames": len(aug),
        "min_clearance": min_clear,
        "ref_min_clearance": float(info.get("ref_min_clearance", 0.0)),
        "root_pos_err": pos_err,
        "root_rot_err": rot_err,
        "pick_err": pick_err,
        "warnings": "; ".join(info.get("warnings", [])) or "-",
        "passed": bool(min_clear >= thr["min_clearance"]
                       and pos_err <= thr["max_root_pos_err"]
                       and rot_err <= thr["max_root_rot_err"]
                       and pick_err <= thr["max_pick_err"]),
    }


def _evaluate(task_dirs) -> list:
    return [(d.name, _evaluate_task(d)) for d in task_dirs]


HEADER = (f"{'task':28s} {'pen mm':>7s} {'penfr':>6s} {'clr mm':>7s} "
          f"{'drift m':>8s} {'rad':>6s} {'pick m':>7s} {'PASS':>5s}  warnings")


def _format_row(name: str, r: dict | None) -> str:
    if r is None:
        return f"{name:28s} (no result)"
    pen_frames = f"{r['pen_frames']}/{r['frames']}"
    return (f"{name:28s} {r['penetration'] * 1000:7.1f} {pen_frames:>6s} "
            f"{r['min_clearance'] * 1000:7.1f} {r['root_pos_err']:8.3f} "
            f"{r['root_rot_err']:6.3f} {r['pick_err']:7.3f} "
            f"{'YES' if r['passed'] else 'no':>5s}  {r['warnings']}")


def _format_table(rows) -> str:
    n_pass = sum(1 for _, r in rows if r and r["passed"])
    n_run = sum(1 for _, r in rows if r)
    lines = [HEADER]
    lines += [_format_row(name, r) for name, r in rows]
    lines.append(f"\ncollision-free + task preserved: {n_pass}/{n_run} run "
                 f"({len(rows)} built)  [{RESULT_NPZ}]")
    return "\n".join(lines)


def _verdict(rows) -> str:
    graded = [r for _, r in rows if r]
    if not graded:
        return "error"
    return "PASS" if any(r["passed"] for r in graded) else "failed"


TASK = Task(
    name="under_table",
    subtree=SUBTREE,
    scene_defaults=SCENE_DEFAULTS,
    solve_defaults=SOLVE_DEFAULTS,
    solve_int_keys=SOLVE_INT_KEYS,
    reconstruct=_reconstruct,
    solve_command=_solve_command,
    evaluate=_evaluate,
    format_table=_format_table,
    format_row=_format_row,
    verdict=_verdict,
    passed=lambda v: v == "PASS",
)
