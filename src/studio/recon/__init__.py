"""Scene reconstruction: a robot-only Kimodo clip -> a physics trial.

This is the science studio owns. It is a port of mppi_locoma's
reconstruction stack (grasp detection, interaction graph, hindsight scene
reconstruction, trial emission — all SceneBot-derived, arXiv 2606.27581),
with one capability that repo has no notion of: forcing the contact
window and re-deriving everything measured from it, so a misdetected
window can be corrected from the GUIs.

Needs numpy + mujoco and the G1 assets (see `assets.py`); no solver and
no GPU, so it runs in studio's own venv.

    qpos, meta, grasp = recon.detect(npz)
    task_dir, grasp, line = recon.reconstruct(npz, out_root, task)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from . import assets
from .grasp import GraspInfo, detect_grasp, override_window
from .graph import build_interaction_graph
from .loader import load_kimodo_constraints, load_kimodo_npz
from .scene import SCENE_DEFAULTS, emit_trial

__all__ = [
    "assets", "GraspInfo", "SCENE_DEFAULTS", "build_interaction_graph",
    "detect", "detect_grasp", "load_kimodo_npz", "load_kimodo_constraints",
    "override_window", "reconstruct",
]


def detect(npz_path: Path, allow_held_start: bool = False):
    """Load a clip and detect its grasp. Returns (qpos, meta, grasp)."""
    qpos, meta = load_kimodo_npz(npz_path, use_smooth_pos=False)
    cons = load_kimodo_constraints(npz_path)
    grasp = detect_grasp(meta, qpos, cons, allow_held_start=allow_held_start)
    return qpos, meta, grasp


def reconstruct(
    npz_path: Path,
    out_root: Path,
    task: str,
    scene_params: Optional[Dict] = None,
    allow_held_start: bool = False,
    pick: Optional[int] = None,
    release: Optional[int] = None,
) -> Tuple[Optional[Path], GraspInfo, str]:
    """Full reconstruction of one clip.

    pick/release force the contact window; either alone keeps the
    detected value for the other. Returns (task_dir or None, grasp,
    one-line summary) — None means the motion is inconsistent with any
    scene, which is a verdict rather than an error.
    """
    params = {**SCENE_DEFAULTS, **(scene_params or {})}
    qpos, meta, grasp = detect(npz_path, allow_held_start)
    if pick is not None or release is not None:
        grasp = override_window(
            meta, grasp,
            grasp.pick_frame if pick is None else pick,
            grasp.release_frame if release is None else release)

    trial = emit_trial(
        qpos, meta, grasp, Path(out_root), task,
        hand_geom=params["hand_geom"],
        box_mass=float(params["box_mass"]),
        box_height=float(params["box_height"]),
        box_depth=float(params["box_depth"]),
        squeeze=float(params["squeeze"]),
        floor_snap_below=float(params["floor_snap_below"]),
    )
    return (trial.parent if trial else None), grasp, _summary(task, grasp, trial)


def _summary(task: str, grasp: GraspInfo, trial: Optional[Path]) -> str:
    """One line carrying everything worth seeing in a log or a GUI."""
    import json

    line = (f"{task}: pick f{grasp.pick_frame} release f{grasp.release_frame} "
            f"box_w {grasp.box_width:.3f} (raw {grasp.raw_carry_gap:.3f}) "
            f"lift {grasp.lift_height:.2f} "
            f"flags [{','.join(grasp.quality_flags) or '-'}]")
    if trial is None:
        return line + (" -> SKIPPED (motion inconsistent with any scene: "
                       "support sweep or unresolvable spawn collision)")

    info = json.loads((trial.parent / "task_info.json").read_text())
    line += f" terrain {len(info['terrain_geoms'])} geoms"
    place = info.get("box_placement") or {}
    shift = place.get("shift", [0.0, 0.0])
    if any(abs(s) > 1e-6 for s in shift) or abs(place.get("dyaw", 0.0)) > 1e-6:
        line += (f" box-shift ({shift[0]:+.2f},{shift[1]:+.2f}m,"
                 f"{np.degrees(place['dyaw']):+.0f}deg,"
                 f" pen {place['max_pen_before'] * 100:.1f}cm cleared)")
    pen = info.get("spawn_penetration") or {}
    if pen:
        line += f" SPAWN-PEN {max(pen.values()) * 1000:.0f}mm ({len(pen)} pairs)"
    return line + f" -> {trial}"
