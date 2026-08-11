"""Retargeting quality for built trials.

numpy only — no torch, no SPIDER — so it runs in studio's own venv and
the verdict never needs the solve runtime.

Primary success is the LIFT: the box must rise at least 70% of the
reference lift and end within 0.2 m of the reference final position.
Tracking errors are secondary diagnostics.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

RESULT_NPZ = "trajectory_mjwp.npz"
REFERENCE_NPZ = "trajectory_kinematic.npz"

# DynaRetarget success thresholds (Eqs. 2-3)
DR_POS_MAX = 0.10          # m, mean object position error
DR_ROT_MAX = np.deg2rad(25.0)
LIFT_MIN_REF = 0.1         # m, below this the reference never really lifts
LIFT_FRACTION = 0.7        # sim must reach this much of the reference lift
LIFT_FINAL_MAX = 0.2       # m, final object position error


def _smoothness(q_joints: np.ndarray) -> float:
    """DynaRetarget Eq. 5: L1 sum of finite-difference joint accelerations.

    The dt^2 factor is omitted — callers only use the sim/ref RATIO on a
    shared timebase, where it cancels.
    """
    if len(q_joints) < 3:
        return 0.0
    acc = q_joints[2:] - 2 * q_joints[1:-1] + q_joints[:-2]
    return float(np.abs(acc).sum())


def evaluate_task(task_dir: Path,
                  result: str = RESULT_NPZ) -> Optional[dict]:
    """Metrics for one built trial, or None if it has no solve result."""
    res_path = task_dir / "0" / result
    if not res_path.exists():
        return None

    q = np.load(res_path)["qpos"].reshape(-1, 43)
    rq = np.load(task_dir / "0" / REFERENCE_NPZ)["qpos"]
    info = json.loads((task_dir / "task_info.json").read_text())

    # the solve runs at 60 Hz, the reference at 30 — index the reference at
    # half rate, clamped to its last frame
    idx = np.minimum(np.arange(len(q)) // 2, len(rq) - 1)
    op = np.linalg.norm(q[:, 36:39] - rq[idx, 36:39], axis=1)
    dot = np.abs(np.sum(q[:, 39:43] * rq[idx, 39:43], axis=1)).clip(0, 1)
    orot = 2 * np.arccos(dot)
    bp = np.linalg.norm(q[:, :3] - rq[idx, :3], axis=1)

    ref_lift = rq[-1, 38] - rq[0, 38]
    sim_lift = q[-1, 38] - rq[0, 38]
    lifted = bool(ref_lift > LIFT_MIN_REF
                  and sim_lift >= LIFT_FRACTION * ref_lift
                  and op[-1] < LIFT_FINAL_MAX)
    succ_dr = bool(op.mean() < DR_POS_MAX and orot.mean() < DR_ROT_MAX)

    # smoothness ratio, sim decimated onto the reference timebase
    qs = q[::2][:len(rq)]
    s_ref = _smoothness(rq[:len(qs), 7:36])
    smooth = _smoothness(qs[:, 7:36]) / max(s_ref, 1e-9)

    return {
        "op_mean": float(op.mean()), "op_final": float(op[-1]),
        "orot_mean": float(orot.mean()), "bp_mean": float(bp.mean()),
        "final_box_z": float(q[-1, 38]), "ref_final_box_z": float(rq[-1, 38]),
        "flags": ",".join(info.get("quality_flags", [])) or "-",
        "lifted": lifted, "succ_dr": succ_dr, "smooth": float(smooth),
    }


def evaluate_tasks(task_dirs, result: str = RESULT_NPZ) -> list[tuple]:
    """[(task name, metrics or None)] for each trial dir."""
    return [(d.name, evaluate_task(d, result)) for d in task_dirs]


HEADER = (f"{'task':28s} {'obj_pos':>8s} {'obj_rot':>8s} {'base':>6s} "
          f"{'boxz f/r':>12s} {'smooth':>6s} {'LIFT':>4s} {'DR':>3s}  flags")


def format_row(name: str, r: Optional[dict]) -> str:
    if r is None:
        return f"{name:28s} (no result)"
    return (f"{name:28s} {r['op_mean']:8.3f} {r['orot_mean']:8.3f} "
            f"{r['bp_mean']:6.3f} "
            f"{r['final_box_z']:5.2f}/{r['ref_final_box_z']:4.2f} "
            f"{r['smooth']:6.2f} "
            f"{'YES' if r['lifted'] else ' no'} "
            f"{'ok' if r['succ_dr'] else 'no':>3s}  {r['flags']}")


def format_table(rows: list[tuple], result: str = RESULT_NPZ) -> str:
    n_ok = sum(1 for _, r in rows if r and r["lifted"])
    n_dr = sum(1 for _, r in rows if r and r["succ_dr"])
    n_run = sum(1 for _, r in rows if r)
    lines = [HEADER]
    lines += [format_row(name, r) for name, r in rows]
    lines.append(f"\nlifted: {n_ok}/{n_run} run, DR-success: {n_dr}/{n_run} "
                 f"({len(rows)} built)  [{result}]")
    return "\n".join(lines)


def verdict(rows: list[tuple]) -> str:
    """The manifest verdict for a run: LIFT beats DR beats failed."""
    graded = [r for _, r in rows if r]
    if not graded:
        return "error"
    if any(r["lifted"] for r in graded):
        return "LIFT"
    if any(r["succ_dr"] for r in graded):
        return "DR"
    return "failed"
