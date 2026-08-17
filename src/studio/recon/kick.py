"""Kick-obstacle scene reconstruction: an obstacle in the swing path.

The kick sibling of `table.py`: detect the kicking foot's swing from FK of
the foot sites, place a floor-standing box (seeded position along the
approach path, lateral/yaw jitter) so the REFERENCE swing penetrates it,
and emit a trial the MPPI solve re-paths around — reference collides,
augmented motion doesn't, and the kick point itself stays reachable.

Placement invariants, in priority order:

1. the reference foot sinks ``max_ref_pen`` below the box top at the
   chosen path point (the closeness knob — this is the solver's work);
2. the kick point stays outside the box by ``kick_margin`` (else avoiding
   the obstacle would forbid the kick itself) — the obstacle slides back
   along the path until it does;
3. everything that is NOT the kicking leg clears the box by
   ``body_clearance`` — the box shifts across the path until it does
   (same philosophy as the table task's arm_conflict: only the limb the
   solver re-paths is allowed to conflict).

numpy + mujoco only (studio's own venv); shares the robot model, SDF and
FK utilities, injection, and trial emission with `table.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np

from .table import (FOOT_GEOMS, _collision_geom_entries, _point_box_sdf_np,
                    emit_obstacle_trial, robot_model,
                    trajectory_collision_points)

# geoms of each leg chain, by side — the kicking side's set is allowed to
# conflict with the obstacle (re-pathing it is the solver's work)
_LEG_KEYWORDS = ("hip", "thigh", "shin", "linkage")
_SIDE_FOOT = {"left": ("lf0", "lf1", "lf2", "lf3"),
              "right": ("rf0", "rf1", "rf2", "rf3")}


@dataclass
class KickSpec:
    """One floor-standing obstacle box in the kick path (world frame)."""

    center: np.ndarray  # (3,)
    half: np.ndarray    # (3,)
    yaw: float = 0.0
    meta: dict = field(default_factory=dict)

    def primitives(self) -> list:
        return [("box", self.center.tolist(), [*self.half.tolist(), self.yaw])]

    def geom_names(self) -> list:
        return ["kick_obstacle_geom"]

    def boxes(self) -> list:
        return [("kick_obstacle_geom", self.center, self.half, self.yaw,
                 "0.65 0.3 0.15 1.0")]


def foot_site_trajectory(
    qpos_ref: np.ndarray, mj_model: mujoco.MjModel
) -> np.ndarray:
    """World foot-site positions (T, 2, 3), left then right."""
    sids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, s)
            for s in ("left_foot", "right_foot")]
    assert -1 not in sids, "foot sites missing — build via robot_model()"
    data = mujoco.MjData(mj_model)
    feet = np.zeros((len(qpos_ref), 2, 3))
    for t in range(len(qpos_ref)):
        data.qpos[:] = qpos_ref[t]
        mujoco.mj_fwdPosition(mj_model, data)
        feet[t, 0] = data.site_xpos[sids[0]]
        feet[t, 1] = data.site_xpos[sids[1]]
    return feet


def kick_leg_point_mask(mj_model: mujoco.MjModel, side: str) -> np.ndarray:
    """Bool mask over trajectory_collision_points(..., FOOT_GEOMS) columns:
    True = the point belongs to the ``side`` leg chain (incl. its foot
    pads) — the limb allowed to conflict with the obstacle."""
    mask = []
    for name, _, _, cap, _ in _collision_geom_entries(mj_model, FOOT_GEOMS):
        leg = (name.startswith(f"{side}_")
               and any(k in name for k in _LEG_KEYWORDS)
               ) or name in _SIDE_FOOT[side]
        mask += [leg, leg, leg] if cap else [leg]
    return np.asarray(mask)


def detect_kick(feet: np.ndarray, lift_thresh: float):
    """(side, window (s0, s1), kick_frame) of the dominant swing.

    side: 0 = left, 1 = right. The window is the contiguous run of frames
    around the highest foot lift where the foot stays above the stance
    height by lift_thresh; the kick frame is the window frame where the
    foot is farthest from its window-start position (covers forward kicks
    whose extreme point is not the apex). Raises ValueError when no frame
    lifts — a verdict, not an error.
    """
    stance_z = np.median(feet[:10, :, 2], axis=0)          # (2,)
    lift = feet[:, :, 2] - stance_z[None, :]
    side = int(np.argmax(lift.max(axis=0)))
    swinging = lift[:, side] > lift_thresh
    if not swinging.any():
        raise ValueError(
            f"no kick found (max foot lift {lift.max():.2f}m < "
            f"lift_thresh {lift_thresh}m) — is this a kick motion?")
    apex = int(np.argmax(lift[:, side]))
    s0 = apex
    while s0 > 0 and swinging[s0 - 1]:
        s0 -= 1
    s1 = apex
    while s1 < len(feet) - 1 and swinging[s1 + 1]:
        s1 += 1
    foot = feet[:, side]
    dist = np.linalg.norm(foot[s0:s1 + 1] - foot[s0], axis=1)
    kick_frame = s0 + int(np.argmax(dist))
    if kick_frame - s0 < 4:
        raise ValueError(
            f"kick approach too short ({kick_frame - s0} frames in window "
            f"{s0}-{s1}) — cannot place an obstacle in the path")
    return side, (s0, s1), kick_frame


def estimate_kick(
    qpos_ref: np.ndarray,
    mj_model: mujoco.MjModel,
    kcfg,
    rng: Optional[np.random.RandomState] = None,
    verbose: bool = True,
) -> KickSpec:
    """Place the obstacle box in the kicking foot's approach path."""
    if rng is None:
        rng = np.random.RandomState()
    warnings: list = []

    feet = foot_site_trajectory(qpos_ref, mj_model)
    side, (s0, s1), kick_frame = detect_kick(feet, kcfg.lift_thresh)
    side_name = ("left", "right")[side]
    foot = feet[:, side]

    points, radii = trajectory_collision_points(qpos_ref, mj_model,
                                                extra_geoms=FOOT_GEOMS)
    body_mask = ~kick_leg_point_mask(mj_model, side_name)
    body_flat = points[:, body_mask].reshape(-1, 3)
    body_radii = np.tile(radii[body_mask], len(points))

    frac = float(np.clip(
        kcfg.path_frac + rng.uniform(-kcfg.frac_jitter, kcfg.frac_jitter),
        0.15, 0.85))
    lateral_off = float(rng.uniform(-kcfg.y_jitter, kcfg.y_jitter))
    yaw_off = float(rng.uniform(-kcfg.yaw_jitter, kcfg.yaw_jitter))

    def build(idx: int, shift: float) -> Tuple[KickSpec, float, float]:
        """Spec at approach index + cross-path shift, with the worst
        non-kicking-body clearance and the kick-point clearance."""
        p = foot[idx]
        d = foot[min(idx + 2, kick_frame)] - foot[max(idx - 2, s0)]
        yaw_path = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d[:2]) \
            > 1e-6 else 0.0
        yaw = yaw_path + yaw_off
        lat = np.array([-np.sin(yaw_path), np.cos(yaw_path)])
        top = max(float(p[2]) + kcfg.max_ref_pen, kcfg.min_box_height)
        center = np.array([
            p[0] + lat[0] * (lateral_off + shift),
            p[1] + lat[1] * (lateral_off + shift),
            top / 2,
        ])
        half = np.array([kcfg.box_depth / 2, kcfg.box_width / 2, top / 2])
        spec = KickSpec(center=center, half=half, yaw=yaw)
        body_clear = float(
            (_point_box_sdf_np(body_flat, center, half, yaw)
             - body_radii).min())
        kick_clear = float(_point_box_sdf_np(
            foot[kick_frame][None], center, half, yaw)[0])
        return spec, body_clear, kick_clear

    # slide the obstacle back along the approach until the kick point
    # itself stays clear (invariant 2)
    n = kick_frame - s0
    idx = s0 + int(round(frac * n))
    while idx > s0 + 2:
        _, _, kick_clear = build(idx, 0.0)
        if kick_clear >= kcfg.kick_margin:
            break
        idx -= 1
    # shift across the path until the rest of the body clears (invariant
    # 3), keeping the path point inside the footprint
    max_shift = kcfg.box_width / 2 - 0.05
    best = None
    for k in range(int(max_shift / kcfg.clear_step) + 1):
        for shift in {k * kcfg.clear_step, -k * kcfg.clear_step}:
            spec, body_clear, kick_clear = build(idx, shift)
            if best is None or body_clear > best[1]:
                best = (spec, body_clear, kick_clear)
            if body_clear >= kcfg.body_clearance:
                best = (spec, body_clear, kick_clear)
                break
        else:
            continue
        break
    spec, body_clear, kick_clear = best
    if body_clear < kcfg.body_clearance:
        warnings.append(
            f"non-kicking body clears the obstacle by only "
            f"{body_clear:.3f}m (< {kcfg.body_clearance}m) at max shift")
    if kick_clear < kcfg.kick_margin:
        warnings.append(
            f"kick point clears the obstacle by only {kick_clear:.3f}m "
            f"(< {kcfg.kick_margin}m) — avoiding it may forbid the kick")

    # how deep the reference foot goes below the box top at the path point
    p = foot[idx]
    foot_conflict = float((spec.center[2] + spec.half[2]) - p[2])
    ref_min_clearance = float(
        (_point_box_sdf_np(points.reshape(-1, 3), spec.center, spec.half,
                           spec.yaw) - np.tile(radii, len(points))).min())

    spec.meta = {
        "side": side_name,
        "window": (int(s0), int(s1)),
        "kick_frame": int(kick_frame),
        "path_index": int(idx),
        "foot_conflict": foot_conflict,
        "kick_clearance": kick_clear,
        "body_clearance": body_clear,
        "ref_min_clearance": ref_min_clearance,
        "warnings": warnings,
    }

    if verbose:
        print(f"  kick: {side_name} foot, window {s0}-{s1}, kick frame "
              f"{kick_frame}, obstacle at path frame {idx}")
        print(f"  box top z={spec.center[2] + spec.half[2]:.3f} "
              f"({kcfg.box_depth:.2f}x{kcfg.box_width:.2f}m, "
              f"yaw={np.degrees(spec.yaw):+.1f}deg), foot conflict "
              f"{foot_conflict:+.3f}m, kick clearance {kick_clear:+.3f}m, "
              f"body clearance {body_clear:+.3f}m")
        for w in warnings:
            print(f"  WARNING: {w}")
    return spec


def reconstruct(
    npz_path: Path,
    out_root: Path,
    task: str,
    params: Dict,
) -> Tuple[Optional[Path], Optional[KickSpec], str]:
    """Full kick-obstacle reconstruction of one clip with resolved
    ``params``. Returns (task_dir or None, spec or None, summary line) —
    None means no kick was found, a verdict rather than an error."""
    from .loader import load_kimodo_npz

    qpos, meta = load_kimodo_npz(npz_path)
    model = robot_model()
    rng = np.random.RandomState(int(params["scene_seed"]))
    try:
        spec = estimate_kick(qpos, model, SimpleNamespace(**params), rng=rng)
    except ValueError as e:
        return None, None, f"{task}: reconstruction SKIPPED ({e})"

    trial = emit_obstacle_trial(
        qpos, meta, spec.boxes(), spec.primitives(), spec.geom_names(),
        Path(out_root), task, params, "kick",
        {"kick_side": spec.meta["side"],
         "kick_frame": spec.meta["kick_frame"],
         **{k: spec.meta[k] for k in
            ("window", "path_index", "foot_conflict", "kick_clearance",
             "body_clearance", "ref_min_clearance", "warnings")}})
    m = spec.meta
    line = (f"{task}: kick {m['side']} f{m['kick_frame']} obstacle top "
            f"z={spec.center[2] + spec.half[2]:.3f} at f{m['path_index']} "
            f"seed {params['scene_seed']} "
            f"foot-conf {m['foot_conflict']:+.3f} "
            f"kick-clr {m['kick_clearance']:+.3f} "
            f"ref-clr {m['ref_min_clearance']:+.3f} "
            f"warn [{'; '.join(m['warnings']) or '-'}] -> {trial}")
    return trial.parent, spec, line
