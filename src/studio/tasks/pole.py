"""Pole interaction: holding or carrying a pole-like object (floor
lamp, tripod, clothes stand) around its real OMOMO mesh, reconstructed
from a hold clip and solved with the same SBMPC loop as box_carry.

The trial contract is identical to box_carry, so the solve command,
evaluator and verdicts are reused unchanged; task_info's ``task_type``
(pole_carry) switches the solve loop onto the per-hand pocket grip
reward. A carried pole barely rises, so DR tracking success is the pass
criterion, with LIFT still counting when a clip does lift.

The hold window, hands, per-hand engagement and wrist retarget are all
detected; object="auto" picks the mesh from the clip's filename. The
left/right scene params override each hand's contact window (off / S-E
frames) — e.g. carry a lamp one-handed from a two-hand clip.
"""

import json
from pathlib import Path

from .. import solve
from ..config import (POLE_SCENE_DEFAULTS, POLE_SOLVE_OVERRIDES,
                      SOLVE_DEFAULTS as _SOLVE_BASE, SOLVE_INT_KEYS,
                      TASK_SUBTREE)
from ..recon import loader, pole
from .base import Task
from . import overrides

# in studio.config (not here) so the MuJoCo-free demo add-on can read it
SCENE_DEFAULTS = POLE_SCENE_DEFAULTS

# same solver and keys as box_carry, pole-tuned values
SOLVE_DEFAULTS = {**_SOLVE_BASE, **POLE_SOLVE_OVERRIDES}


def _merge(*layers) -> dict:
    out = dict(SCENE_DEFAULTS)
    for layer in layers:
        for key, val in (layer or {}).items():
            if key not in out:
                raise SystemExit(f"unknown pole parameter: {key}")
            out[key] = type(out[key])(val)
    return out


def _reconstruct(npz, out_root, task, options):
    options = options or {}
    params = _merge(overrides("pole", "scene"),
                    options.get("scene_params"))
    qpos, meta = loader.load_kimodo_npz(npz, use_smooth_pos=False)
    info = pole.detect_pole_hold(meta, qpos)
    info = pole.apply_hand_windows(info, meta, qpos,
                                   left=str(params["left"]),
                                   right=str(params["right"]))
    obj = pole.infer_object(Path(npz).stem, str(params["object"]))

    # a hand joining late or releasing early shows its own window
    hands = "+".join(
        h + ("" if g == info.grab_frame and e == info.end_frame
             else f"@f{g}-f{e}")
        for h, g, e in zip(info.hands, info.hand_grabs, info.ends))
    line = (f"{task}: {hands} hold f{info.grab_frame}"
            f"-f{info.end_frame} {obj} "
            f"transport {info.transport:.2f} "
            f"flags [{','.join(info.quality_flags) or '-'}]")
    if not info.ok:
        return None, line + " -> SKIPPED (no pole hold)"

    trial = pole.emit_pole_trial(
        qpos, meta, info, Path(out_root), task,
        object_name=obj,
        mass=float(params["mass"]),
        grasp_close=float(params["grasp_close"]),
        grasp=str(params["grasp"]),
    )
    if trial is None:
        return None, line + " -> SKIPPED (spawn collision)"
    clear = (json.loads((trial.parent / "task_info.json").read_text())
             .get("palm_clearance") or {})
    if clear:
        line += (f" palm-clear (pen {clear['max_pen_before_mm']:.0f}mm "
                 f"cleared, shift {clear['max_shift_mm']:.0f}mm)")
    return trial.parent, line + f" -> {trial}"


TASK = Task(
    name="pole",
    subtree=TASK_SUBTREE,
    scene_defaults=SCENE_DEFAULTS,
    solve_defaults=SOLVE_DEFAULTS,
    solve_int_keys=SOLVE_INT_KEYS,
    reconstruct=_reconstruct,
    solve_command=solve.command,
    evaluate=solve.evaluate.evaluate_tasks,
    format_table=solve.evaluate.format_table,
    format_row=solve.evaluate.format_row,
    verdict=solve.evaluate.verdict,
    passed=lambda v: v in ("LIFT", "DR"),
)
