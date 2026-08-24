"""Ground pick: one-hand pick-up of a small object (cube / cylinder /
ball) from the floor, reconstructed from a single-hand reach clip and
solved with the same SBMPC loop as box_carry.

The trial contract is identical to box_carry, so the solve command,
evaluator and verdicts are reused unchanged; task_info's ``task_type:
ground_pick`` switches the solve loop off the bimanual box-face reward.
shape="auto" reads cube/cylinder/ball from the clip's filename; a run
holds one shape, so reconstruct again with --set shape=... for others.
"""

from pathlib import Path

from .. import solve
from ..config import (PICK_SCENE_DEFAULTS, SOLVE_DEFAULTS, SOLVE_INT_KEYS,
                      TASK_SUBTREE)
from ..recon import loader, pick
from .base import Task
from . import overrides

# in studio.config (not here) so the MuJoCo-free demo add-on can read it
SCENE_DEFAULTS = PICK_SCENE_DEFAULTS


def _merge(*layers) -> dict:
    out = dict(SCENE_DEFAULTS)
    for layer in layers:
        for key, val in (layer or {}).items():
            if key not in out:
                raise SystemExit(f"unknown ground_pick parameter: {key}")
            out[key] = val if key == "shape" else type(out[key])(val)
    return out


def _reconstruct(npz, out_root, task, options):
    options = options or {}
    params = _merge(overrides("ground_pick", "scene"),
                    options.get("scene_params"))
    qpos, meta = loader.load_kimodo_npz(npz, use_smooth_pos=False)
    info = pick.detect_pick(meta)
    shape = pick.infer_shape(Path(npz).stem, str(params["shape"]))

    line = (f"{task}: {info.hand} hand pick f{info.pick_frame} "
            f"end f{info.end_frame} reach {info.reach_height:.2f} "
            f"lift {info.lift_height:.2f} {shape} "
            f"flags [{','.join(info.quality_flags) or '-'}]")
    if not info.ok:
        return None, line + " -> SKIPPED (no clean one-hand pick)"

    trial = pick.emit_pick_trial(
        qpos, meta, info, Path(out_root), task,
        shape=shape,
        size=float(params["size"]),
        cyl_height=float(params["cyl_height"]),
        mass=float(params["mass"]),
        grasp_close=float(params["grasp_close"]),
        grasp=str(params["grasp"]),
    )
    if trial is None:
        return None, line + " -> SKIPPED (spawn collision)"
    return trial.parent, line + f" -> {trial}"


TASK = Task(
    name="ground_pick",
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
    passed=lambda v: v == "LIFT",
)
