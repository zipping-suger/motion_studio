"""The reconstruction pipeline every task runs through, and the hooks a
task supplies to it.

    clip  = load_clip(npz)                       # the Kimodo-format npz
    inter = task.detect(qpos, meta, params, options)   # when / which hands
    built = task.build(qpos, meta, inter, params)      # object, its path,
                                                       # the grasp, the scene
    ---- shared from here (emit) ----
    spawn veto      frame-0 robot/scene penetration beyond SPAWN_SKIP_PEN
    contact ref     each grip's window over its palm's FK track (unless
                    the task supplied its own targets, e.g. box faces)
    graph           SceneBot key-link labels of the window
    write_trial     the files + task_info.json (spec.py)
    describe        the task's extra summary from the written task_info

Adding a task means one `ReconTask` (a detector, an object, a
trajectory, a grasp — composed from the shared kernels: `mjcf` for the
scene, `pick.calibrate_grasp` / `pick.wrist_ik_frame` /
`pole._conform_fingers` for the hands, `spec.assemble_qpos` and
friends) plus its `studio.tasks` glue; nothing here or in the solver
changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Protocol, Tuple

import numpy as np

from .loader import load_kimodo_npz
from .spec import (SPAWN_SKIP_PEN, Built, Interaction, Skip, default_contact,
                   default_graph, palm_site_ids, palm_tracks, spawn_check,
                   write_trial)


class ReconTask(Protocol):
    name: str

    def detect(self, qpos: np.ndarray, meta: Dict, params: Dict,
               options: Dict) -> Interaction: ...

    def build(self, qpos: np.ndarray, meta: Dict, inter: Interaction,
              params: Dict) -> Built: ...

    def describe(self, task_info: Dict) -> str: ...   # optional


def load_clip(npz: Path):
    """(qpos (T, 36) body contract, meta) of a Kimodo-format clip."""
    return load_kimodo_npz(npz, use_smooth_pos=False)


def emit(task: ReconTask, qpos: np.ndarray, meta: Dict, inter: Interaction,
         out_root: Path, name: str, params: Dict,
         data_id: int = 0) -> Tuple[Optional[Path], Optional[Dict], str]:
    """Build + write one trial -> (trial dir, task_info, skip reason)."""
    try:
        built = task.build(qpos, meta, inter, params)
    except Skip as e:
        return None, None, str(e)

    spawn_pen, hard = spawn_check(built.model, built.qpos[0],
                                  built.spec.starts_held)
    if hard and max(hard.values()) > SPAWN_SKIP_PEN:
        return None, None, "spawn collision"
    if hard:
        built.spec.flags.append("spawn_penetration")

    T = len(built.qpos)
    site_ids = palm_site_ids(built.model)
    contact = (built.contact if built.contact is not None
               else default_contact(built.spec, T))
    contact_pos = (built.contact_pos if built.contact_pos is not None
                   else palm_tracks(built.model, built.qpos, site_ids))
    graph = built.graph if built.graph is not None \
        else default_graph(meta, built.spec)
    trial, info = write_trial(out_root, name, built, meta, site_ids, contact,
                              contact_pos, graph, spawn_pen, data_id=data_id)
    return trial, info, ""


def reconstruct(task: ReconTask, npz: Path, out_root: Path, name: str,
                params: Dict, options: Optional[Dict] = None
                ) -> Tuple[Optional[Path], str]:
    """Full reconstruction of one clip -> (task dir or None, one-line
    summary). None means the motion admits no scene of this family (the
    line says why), a verdict rather than an error."""
    qpos, meta = load_clip(npz)
    inter = task.detect(qpos, meta, params, options or {})
    line = f"{name}: {inter.detail}"
    if not inter.ok:
        return None, line + f" -> SKIPPED ({inter.skip})"
    trial, info, why = emit(task, qpos, meta, inter, out_root, name, params)
    if trial is None:
        return None, line + f" -> SKIPPED ({why})"
    describe = getattr(task, "describe", None)
    if describe is not None:
        line += describe(info)
    return trial.parent, line + f" -> {trial}"
