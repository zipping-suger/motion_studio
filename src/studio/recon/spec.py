"""The reconstruction's intermediate representation and the trial it
becomes — the recon -> solve contract, written by one function.

Every downstream task is a human-object interaction reconstructed from
a robot-only clip: detect WHEN and with WHICH hands the robot touches
the object, decide WHAT the object is and WHERE it goes, pose the hands
on it, and emit a physics trial. The task-specific parts of that (the
detector, the object, its trajectory, the grasp) produce these
records; `write_trial` turns them into the files SPIDER's solve reads:

    <out_root>/processed/kimodo/unitree_g1/humanoid_object/<task>/
        scene.xml                     the MuJoCo scene: robot + object
        task_info.json                TrialSpec, serialized (below)
        <data_id>/trajectory_kinematic.npz
            qpos/qvel/ctrl            the reference, model layout, object
                                      free joint LAST (layout.check_scene)
            contact/contact_pos       per-palm hold mask + target point
            link_contact/link_pos     SceneBot key-link labels (graph)

task_info.json keys every trial writes (`studio.solve.spec` reads the
solver's subset; the GUIs read `object` and `terrain_geoms`):

    task_type, ref_dt, contact_site_ids, source_npz,
    pick_frame, release_frame (the interaction window; null = none),
    starts_held, lift_height,
    object: {kind, faces, symmetry, size, mass, ...task fields},
    grips: [{hand, grab_frame, release_frame, pocket, anchor, ...}],
    grasp: [per-hand retarget report], supports, hand_noise_scale,
    quality_flags, spawn_penetration,
    key_links, scene_types, graph_flags, terrain_geoms,
    plus the task's own `TrialSpec.extra` at the top level.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from ..config import TASK_SUBTREE
from . import layout
from .graph import KEY_LINKS, SCENE_TYPES, InteractionGraph, build_interaction_graph
from .grasp import GraspInfo
from .loader import compute_qvel

PALM_SITES = ("left_palm", "right_palm")
PALM_COLUMN = {"left": 0, "right": 1}

# --- spawn check -----------------------------------------------------------
ROBOT_CONTACT_GEOMS = ("lh", "rh", "lf0", "lf1", "lf2", "lf3",
                       "rf0", "rf1", "rf2", "rf3")
SPAWN_PEN_TOL = 0.004   # m: report robot-vs-scene penetrations deeper than this
SPAWN_SKIP_PEN = 0.03   # m: frame-0 penetration beyond this -> skip the clip
                        # (placement search maxed out; settle would explode)
# hand/wrist geoms whose frame-0 object contact is expected for held starts
HAND_SPAWN_GEOMS = ("lh", "rh", "left_wrist_collision", "right_wrist_collision")
_OBJECT_GEOM = re.compile(r"largebox_geom$|largebox_c\d+$")


class Skip(Exception):
    """The motion admits no scene of this family: a verdict, not an
    error. The message is the reason the summary line shows."""


@dataclass
class Interaction:
    """What `ReconTask.detect` found."""
    ok: bool
    detail: str                   # the summary line's body
    skip: str = ""                # why not ok
    info: Any = None              # the task's own detection record
    object: Optional[str] = None  # the resolved object (shape / mesh name)


@dataclass
class ObjectSpec:
    kind: str                     # box | primitive | mesh
    size: List[float]             # bounding dims, for viewers to frame it
    mass: float
    faces: bool = False           # a box whose palm-side faces are the
                                  # contact targets (the face reward)
    symmetry: str = "none"        # "axial": roll about its axis is not data
    info: Dict = field(default_factory=dict)   # task fields (shape, name...)

    def as_dict(self) -> Dict:
        return {"kind": self.kind, "faces": bool(self.faces),
                "symmetry": self.symmetry,
                "size": [round(float(v), 4) for v in self.size],
                "mass": float(self.mass), **self.info}


@dataclass
class Grip:
    """One holding hand: its window, the calibrated palm pocket the
    solve's grip reward holds the object in, and the object-frame anchor
    it holds there (None = the object's origin)."""
    hand: str
    window: Tuple[int, int]
    pocket: Optional[Any]
    anchor: Optional[Any] = None
    extra: Dict = field(default_factory=dict)

    @property
    def column(self) -> int:
        return PALM_COLUMN[self.hand]

    def as_dict(self) -> Dict:
        return {
            "hand": self.hand,
            "grab_frame": int(self.window[0]),
            "release_frame": int(self.window[1]),
            "pocket": ([round(float(v), 4) for v in self.pocket]
                       if self.pocket is not None else None),
            "anchor": ([round(float(v), 4) for v in self.anchor]
                       if self.anchor is not None else None),
            **self.extra,
        }


@dataclass
class TrialSpec:
    task_type: str
    object: ObjectSpec
    window: Tuple[Optional[int], Optional[int]] = (None, None)
    starts_held: bool = False
    lift_height: float = 0.0
    grips: List[Grip] = field(default_factory=list)
    grasp: List[Dict] = field(default_factory=list)     # retarget reports
    supports: List[Dict] = field(default_factory=list)
    terrain: List[Dict] = field(default_factory=list)   # static boxes
    flags: List[str] = field(default_factory=list)      # quality_flags
    graph_flags: List[str] = field(default_factory=list)
    hand_noise_scale: Optional[float] = None
    extra: Dict = field(default_factory=dict)


@dataclass
class Built:
    """What `ReconTask.build` returns: the scene and the reference in
    it. contact / contact_pos / graph default from the spec (each grip's
    window over its palm's own FK track; the interaction graph of the
    window) when None; pad appends that many frozen frames, truncate
    cuts the reference first."""
    scene_xml: str
    model: mujoco.MjModel
    qpos: np.ndarray
    spec: TrialSpec
    contact: Optional[np.ndarray] = None
    contact_pos: Optional[np.ndarray] = None
    graph: Optional[InteractionGraph] = None
    pad: int = 0
    truncate: Optional[int] = None


# --------------------------------------------------------------- helpers --

def trial_paths(out_root: Path, task: str, data_id: int = 0) -> Tuple[Path, Path]:
    task_dir = Path(out_root) / TASK_SUBTREE / task
    return task_dir, task_dir / str(data_id)


def assemble_qpos(model, qpos_robot: np.ndarray, obj_qpos: np.ndarray) -> np.ndarray:
    """Model-layout qpos: base + body joints at their addresses (hand
    joints rest at 0), the object free joint last."""
    qpos = np.zeros((len(qpos_robot), model.nq))
    qpos[:, :7] = qpos_robot[:, :7]
    qpos[:, layout.body_addr(model)] = qpos_robot[:, 7:]
    qpos[:, -7:] = obj_qpos
    return qpos


def palm_site_ids(model) -> List[int]:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
           for s in PALM_SITES]
    assert -1 not in ids, "palm sites missing from scene"
    return ids


def palm_tracks(model, qpos: np.ndarray, site_ids: List[int]) -> np.ndarray:
    """World palm-site positions (T, 2, 3) by FK of the reference."""
    data = mujoco.MjData(model)
    out = np.zeros((len(qpos), 2, 3))
    for t in range(len(qpos)):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        out[t, 0] = data.site_xpos[site_ids[0]]
        out[t, 1] = data.site_xpos[site_ids[1]]
    return out


def measure_spawn_penetration(model, data) -> Dict[str, float]:
    """Signed-distance check of every robot collision geom against the
    object and terrain geoms at the current state (mj_forward first) ->
    {"<robot_geom>~<scene_geom>": depth} beyond tolerance."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
             for g in range(model.ngeom)]
    robot = [g for g, n in enumerate(names)
             if n.endswith("_collision") or n in ROBOT_CONTACT_GEOMS]
    scene = [g for g, n in enumerate(names)
             if n == "table" or re.match(r"(table|terrain|seat)_\d+$", n)
             or _OBJECT_GEOM.search(n)]
    fromto = np.zeros(6)
    pens: Dict[str, float] = {}
    for gr in robot:
        for gs in scene:
            d = mujoco.mj_geomDistance(model, data, gr, gs, 0.0, fromto)
            if d < -SPAWN_PEN_TOL:
                pens[f"{names[gr]}~{names[gs]}"] = round(float(-d), 4)
    return pens


def spawn_check(model, qpos0: np.ndarray, starts_held: bool):
    """(all frame-0 penetrations, the ones that count). Geometric, NOT
    via data.contact: explicit-pair scenes produce no MuJoCo contact for
    a shin through a slab while still being broken. Holding at spawn
    means palm-vs-object contact by design, so those never count."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)
    pens = measure_spawn_penetration(model, data)
    if not starts_held:
        return pens, pens
    hard = {k: v for k, v in pens.items()
            if not (k.split("~")[0] in HAND_SPAWN_GEOMS
                    and _OBJECT_GEOM.search(k.split("~")[1]))}
    return pens, hard


def default_contact(spec: TrialSpec, T: int) -> np.ndarray:
    """Each grip's palm column, 1 over its window."""
    contact = np.zeros((T, 2))
    for g in spec.grips:
        contact[g.window[0]:g.window[1] + 1, g.column] = 1.0
    return contact


def default_graph(meta: Dict, spec: TrialSpec) -> InteractionGraph:
    """The interaction graph of the spec's window (feet/pelvis-terrain
    edges are detected from the clip; the hands-object edge is the
    window)."""
    s, e = spec.window
    grasp = None
    if s is not None and e is not None:
        w = float(spec.object.size[0])
        grasp = GraspInfo(pick_frame=int(s), release_frame=int(e),
                          box_width=w, raw_carry_gap=w,
                          lift_height=float(spec.lift_height),
                          quality_flags=list(spec.flags),
                          starts_held=bool(spec.starts_held))
    return build_interaction_graph(meta, grasp)


# ---------------------------------------------------------------- write --

def write_trial(out_root: Path, task: str, built: Built, meta: Dict,
                site_ids: List[int], contact: np.ndarray,
                contact_pos: np.ndarray, graph: InteractionGraph,
                spawn_pen: Dict[str, float],
                data_id: int = 0) -> Tuple[Path, Dict]:
    """Write the trial files. Returns (trial dir, the task_info written)."""
    spec = built.spec
    fps = float(meta["fps"])
    task_dir, trial_dir = trial_paths(out_root, task, data_id)
    trial_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "scene.xml").write_text(built.scene_xml)

    qpos = built.qpos
    qvel = compute_qvel(built.model, qpos, 1.0 / fps)
    ctrl = layout.ctrl_reference(built.model, qpos)

    # a trial that ends held is padded with frozen final frames (zero
    # velocity), so the receding horizon has a full reference window at
    # every evaluated frame
    Tt, pad = built.truncate, int(built.pad)

    def cut(a, pad_tail=None):
        out = a if Tt is None else a[:Tt]
        if pad:
            tail = (np.repeat(out[-1:], pad, axis=0)
                    if pad_tail is None else pad_tail)
            out = np.concatenate([out, tail], axis=0)
        return out

    np.savez(
        trial_dir / "trajectory_kinematic.npz",
        qpos=cut(qpos),
        qvel=cut(qvel, np.zeros((pad, qvel.shape[1]))),
        ctrl=cut(ctrl),
        contact=cut(contact), contact_pos=cut(contact_pos),
        # SceneBot-style per-link labels: KEY_LINKS x {terrain, object}
        link_contact=cut(graph.link_contact).astype(np.float32),
        link_pos=cut(graph.link_pos).astype(np.float32),
    )

    s, e = spec.window
    info = {
        "task_type": spec.task_type,
        "ref_dt": 1.0 / fps,
        "contact_site_ids": list(site_ids),
        "source_npz": meta["file_path"],
        "pick_frame": None if s is None else int(s),
        "release_frame": None if e is None else int(e),
        "starts_held": bool(spec.starts_held),
        "lift_height": round(float(spec.lift_height), 4),
        "object": spec.object.as_dict(),
        "grips": [g.as_dict() for g in spec.grips],
        "grasp": list(spec.grasp),
        "supports": list(spec.supports),
        "quality_flags": list(spec.flags),
        "spawn_penetration": spawn_pen,
        # interaction-graph / terrain reconstruction (general motion)
        "key_links": list(KEY_LINKS),
        "scene_types": list(SCENE_TYPES),
        "graph_flags": list(graph.flags) + list(spec.graph_flags),
        "terrain_geoms": list(spec.terrain),
        **spec.extra,
    }
    if spec.hand_noise_scale is not None:
        info["hand_noise_scale"] = float(spec.hand_noise_scale)
    (task_dir / "task_info.json").write_text(json.dumps(info, indent=2))
    return trial_dir, info
