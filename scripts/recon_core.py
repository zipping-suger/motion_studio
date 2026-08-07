"""Scene reconstruction core: mppi_locoma as a LIBRARY, plus a manual
contact-window override that repo has no notion of.

Replaces build_trial.py in the studio flow (single-clip runs only; the
science — detection, object trajectory, terrain, trial emission — is
still entirely mppi_locoma's code). The added capability: force the
grasp window (pick/release frames) and re-measure everything derived
from it (box width from the forced window's median gap, lift, held
start), so a misdetected window can be corrected from the GUIs.

Run with mppi_locoma's python. Importable in-process (panel) or as a
CLI (pipeline/demo add-on subprocess):
    recon_core.py <npz> --out-root R --task T [--pick-frame N]
                  [--release-frame N] [--allow-held-start] [--detect-only]
                  [--box-mass ... scene params]
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

STUDIO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDIO_ROOT / "src"))
from studio.config import load_config  # noqa: E402

CFG = load_config()
sys.path.insert(0, str(CFG.mppi_locoma / "src"))

from grasp_detect import (  # noqa: E402
    BOX_WIDTH_MAX, BOX_WIDTH_MIN, HAND_SURFACE_OFFSET, GraspInfo,
    _smooth, detect_grasp,
)
from kimodo_loader import (  # noqa: E402
    KIM_LEFT_HAND_TIP, KIM_RIGHT_HAND_TIP,
    load_kimodo_constraints, load_kimodo_npz,
)
from repo_config import SCENE_DEFAULTS  # noqa: E402
from scene_recon import emit_trial  # noqa: E402


def detect(npz_path: Path, allow_held_start: bool = False):
    """Load + detect. Returns (qpos, meta, grasp)."""
    qpos, meta = load_kimodo_npz(npz_path, use_smooth_pos=False)
    cons = load_kimodo_constraints(npz_path)
    grasp = detect_grasp(meta, qpos, cons, allow_held_start=allow_held_start)
    return qpos, meta, grasp


def override_grasp_window(meta: dict, grasp: GraspInfo,
                          pick: int, release: int) -> GraspInfo:
    """Force the contact window and RE-MEASURE what depends on it: box
    width from the forced window's median gap, lift from the forced pick,
    held start iff pick is frame 0. Detector-history flags are dropped —
    the window is authoritative now."""
    jp = meta["joint_positions"]
    lh, rh = jp[:, KIM_LEFT_HAND_TIP], jp[:, KIM_RIGHT_HAND_TIP]
    T = len(jp)
    pick = int(np.clip(pick, 0, T - 2))
    release = int(np.clip(release, pick + 1, T - 1))

    gap = np.linalg.norm(lh - rh, axis=1)
    midh = _smooth(0.5 * (lh[:, 2] + rh[:, 2]))
    win = np.arange(pick, release + 1)
    carry_gap = float(np.median(gap[win]))
    surface_gap = carry_gap - 2 * HAND_SURFACE_OFFSET
    box_width = float(np.clip(surface_gap, BOX_WIDTH_MIN, BOX_WIDTH_MAX))

    flags = ["window_override"]
    if pick == 0:
        flags.append("starts_held")
    if float(np.std(gap[win])) > 0.05:
        flags.append("unstable_gap")
    if abs(box_width - surface_gap) > 0.02:
        flags.append("width_clamped")
    lift_height = float(midh[pick:].max() - midh[pick])
    if lift_height < 0.15:
        flags.append("no_lift")

    return dataclasses.replace(
        grasp, pick_frame=pick, release_frame=release, box_width=box_width,
        raw_carry_gap=carry_gap, lift_height=lift_height,
        quality_flags=flags, starts_held=(pick == 0))


def reconstruct(npz_path: Path, out_root: Path, task: str,
                scene_params: dict | None = None,
                allow_held_start: bool = False,
                pick: int | None = None, release: int | None = None):
    """Full reconstruction of one clip. Returns (task_dir_or_None, grasp,
    summary_line)."""
    params = {**SCENE_DEFAULTS, **(scene_params or {})}
    qpos, meta, grasp = detect(npz_path, allow_held_start)
    if pick is not None or release is not None:
        grasp = override_grasp_window(
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
    line = (f"{task}: pick f{grasp.pick_frame} release f{grasp.release_frame} "
            f"box_w {grasp.box_width:.3f} (raw {grasp.raw_carry_gap:.3f}) "
            f"lift {grasp.lift_height:.2f} "
            f"flags [{','.join(grasp.quality_flags) or '-'}]")
    if trial is None:
        line += (" -> SKIPPED (motion inconsistent with any scene: support "
                 "sweep or unresolvable spawn collision)")
        return None, grasp, line
    info = json.loads((trial.parent / "task_info.json").read_text())
    line += f" terrain {len(info['terrain_geoms'])} geoms"
    place = info.get("box_placement") or {}
    sh = place.get("shift", [0.0, 0.0])
    if any(abs(s) > 1e-6 for s in sh) or abs(place.get("dyaw", 0.0)) > 1e-6:
        line += (f" box-shift ({sh[0]:+.2f},{sh[1]:+.2f}m,"
                 f"{np.degrees(place['dyaw']):+.0f}deg,"
                 f" pen {place['max_pen_before'] * 100:.1f}cm cleared)")
    pen = info.get("spawn_penetration") or {}
    if pen:
        line += f" SPAWN-PEN {max(pen.values()) * 1000:.0f}mm ({len(pen)} pairs)"
    line += f" -> {trial}"
    return trial.parent, grasp, line


def main() -> None:
    d = SCENE_DEFAULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--out-root", type=Path, required=False)
    ap.add_argument("--task", required=False)
    ap.add_argument("--detect-only", action="store_true",
                    help="print the detected window as JSON and exit")
    ap.add_argument("--allow-held-start", action="store_true")
    ap.add_argument("--pick-frame", type=int, default=None)
    ap.add_argument("--release-frame", type=int, default=None)
    ap.add_argument("--box-mass", type=float, default=d["box_mass"])
    ap.add_argument("--box-height", type=float, default=d["box_height"])
    ap.add_argument("--box-depth", type=float, default=d["box_depth"])
    ap.add_argument("--squeeze", type=float, default=d["squeeze"])
    ap.add_argument("--hand-geom", choices=["capsule", "mesh"],
                    default=d["hand_geom"])
    ap.add_argument("--floor-snap-below", type=float,
                    default=d.get("floor_snap_below", 0.0))
    args = ap.parse_args()

    if args.detect_only:
        qpos, meta, grasp = detect(args.npz, args.allow_held_start)
        print(json.dumps({
            "n_frames": int(meta["n_frames"]),
            "pick_frame": grasp.pick_frame,
            "release_frame": grasp.release_frame,
            "box_width": grasp.box_width,
            "starts_held": grasp.starts_held,
            "quality_flags": grasp.quality_flags,
        }))
        return
    if args.out_root is None or args.task is None:
        ap.error("--out-root and --task are required unless --detect-only")

    scene_params = {
        "box_mass": args.box_mass, "box_height": args.box_height,
        "box_depth": args.box_depth, "squeeze": args.squeeze,
        "hand_geom": args.hand_geom,
        "floor_snap_below": args.floor_snap_below,
    }
    task_dir, _, line = reconstruct(
        args.npz, args.out_root, args.task, scene_params,
        allow_held_start=args.allow_held_start,
        pick=args.pick_frame, release=args.release_frame)
    print(line)
    sys.exit(0 if task_dir is not None else 1)


if __name__ == "__main__":
    main()
