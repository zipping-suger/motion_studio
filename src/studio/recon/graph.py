"""Scene-interaction graph construction (SceneBot, arXiv 2606.27581, Alg. 1
Stage 2), generalized beyond the bimanual table pick.

The robot's key links K = {left-wrist, right-wrist, left-foot, right-foot,
pelvis} each get candidate contact edges wherever the link moves with low
velocity AND low acceleration relative to a scene node:

  - feet / pelvis vs terrain: the terrain is static, so the criterion is
    world-frame stationarity. For feet it is intersected with Kimodo's
    foot_contacts labels when present (they agree with stationarity on all
    inspected clips and cheaply reject toe-drag frames).
  - wrists vs object: the object is itself inferred from the hands, so the
    criterion becomes rigid co-movement of the hand pair — exactly what
    grasp.detect_grasp already finds. Its window is taken as the
    wrist->object interaction interval.

Terrain support heights are measured against a per-clip floor calibration:
every Kimodo motion starts standing, so the first CAL_FRAMES frames define
the keypoint-to-sole offsets and the z=0 floor. Heights within FLOOR_EPS of
the floor map to the scene's ground plane and produce no edge; only elevated
supports (stairs, ledges, seats) become terrain edges for Stage 3.

Edges violating force closure are flagged upstream (grasp quality flags);
edges whose reconstructed plateau collides with the robot outside the
interaction interval are pruned geometrically by the carving step in
scene.py (Alg. 1 line 27).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .grasp import GraspInfo
from .loader import (
    FPS,
    KIM_LEFT_ANKLE,
    KIM_LEFT_HAND_TIP,
    KIM_LEFT_TOE,
    KIM_PELVIS,
    KIM_RIGHT_ANKLE,
    KIM_RIGHT_HAND_TIP,
    KIM_RIGHT_TOE,
)
from .signal import smooth3

KEY_LINKS = ("left_wrist", "right_wrist", "left_foot", "right_foot", "pelvis")
SCENE_TYPES = ("terrain", "object")

FLOOR_EPS = 0.03       # m: supports closer than this to z=0 are the floor
TERRAIN_MIN_H = 0.06   # m: lowest believable elevated support (real steps);
                       # 3-6 cm "supports" are swing-apex noise -> floor
HEIGHT_STD_MAX = 0.015  # m: a planted foot does not bob
CAL_FRAMES = 10        # frames of the initial stance used for floor calib
FOOT_VEL_MAX = 0.25    # m/s: planted-foot keypoint speed bound
FOOT_ACC_MAX = 2.5     # m/s^2
MIN_CONTACT = 5        # frames: shortest terrain contact worth an edge
BRIDGE = 3             # frames: mask dropouts bridged within one contact
HEIGHT_JUMP = 0.03     # m: a support-height change splits the contact
SIT_MAX_Z = 0.55       # m: pelvis below this can be seated support
SIT_MIN_FRAMES = 15
PELVIS_VEL_MAX = 0.15  # m/s
PELVIS_SEAT_DROP = 0.12  # m: seat surface below the pelvis keypoint
SIT_FEET_OFFSET_MIN = 0.15  # m: seated feet rest forward of the pelvis;
                            # closer means a squat, which needs no seat


@dataclass
class ContactEdge:
    link: str            # one of KEY_LINKS
    scene: str           # "terrain" | "object"
    start: int           # frame interval, inclusive
    end: int
    pos: np.ndarray      # (3,) mean world location; z = support top (terrain)
    height: float = 0.0  # terrain support top, floor-calibrated


@dataclass
class InteractionGraph:
    edges: List[ContactEdge]
    link_contact: np.ndarray   # (T, 5, 2) KEY_LINKS x SCENE_TYPES, 0/1
    link_pos: np.ndarray       # (T, 5, 3) key-link world positions
    flags: List[str] = field(default_factory=list)

    @property
    def terrain_edges(self) -> List[ContactEdge]:
        return [e for e in self.edges if e.scene == "terrain"]


def _stationary(p: np.ndarray, vel_max: float, acc_max: float) -> np.ndarray:
    ps = smooth3(p)
    v = np.gradient(ps, axis=0) * FPS
    a = np.gradient(v, axis=0) * FPS
    return (np.linalg.norm(v, axis=1) < vel_max) & (
        np.linalg.norm(a, axis=1) < acc_max
    )


def _contact_runs(
    mask: np.ndarray, height: np.ndarray, min_len: int
) -> List[Tuple[int, int]]:
    """Connected True-intervals, bridging dropouts <= BRIDGE frames but never
    across a support-height change (stepping up must split the contact)."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    runs = []
    s = e = int(idx[0])
    for t in idx[1:]:
        if t - e <= BRIDGE and abs(height[t] - height[e]) < HEIGHT_JUMP:
            e = int(t)
        else:
            runs.append((s, e))
            s = e = int(t)
    runs.append((s, e))
    return [(s, e) for s, e in runs if e - s + 1 >= min_len]


def build_interaction_graph(
    meta: Dict, grasp: Optional[GraspInfo] = None
) -> InteractionGraph:
    """Stage 2 of SceneBot Alg. 1: candidate edges for every key link."""
    jp = meta["joint_positions"]
    T = len(jp)
    fc = meta.get("foot_contacts")
    flags: List[str] = []
    edges: List[ContactEdge] = []
    link_contact = np.zeros((T, len(KEY_LINKS), len(SCENE_TYPES)))
    link_pos = np.zeros((T, len(KEY_LINKS), 3))

    link_pos[:, 0] = jp[:, KIM_LEFT_HAND_TIP]
    link_pos[:, 1] = jp[:, KIM_RIGHT_HAND_TIP]
    link_pos[:, 4] = jp[:, KIM_PELVIS]

    # --- feet -> terrain -------------------------------------------------
    feet = {
        "left_foot": (KIM_LEFT_ANKLE, KIM_LEFT_TOE, (0, 1)),
        "right_foot": (KIM_RIGHT_ANKLE, KIM_RIGHT_TOE, (2, 3)),
    }
    sole_offsets = {}
    for link, (ankle_i, toe_i, fc_cols) in feet.items():
        ankle, toe = jp[:, ankle_i], jp[:, toe_i]
        # floor calibration: the clip starts standing on z=0 ground
        off_a = float(np.median(ankle[:CAL_FRAMES, 2]))
        off_t = float(np.median(toe[:CAL_FRAMES, 2]))
        sole_offsets[link] = (off_a, off_t)
        height = np.minimum(ankle[:, 2] - off_a, toe[:, 2] - off_t)

        li = KEY_LINKS.index(link)
        link_pos[:, li, :2] = 0.5 * (ankle[:, :2] + toe[:, :2])
        link_pos[:, li, 2] = height

        mask = _stationary(ankle, FOOT_VEL_MAX, FOOT_ACC_MAX) & _stationary(
            toe, FOOT_VEL_MAX, FOOT_ACC_MAX
        )
        if fc is not None:
            mask &= fc[:, fc_cols[0]] | fc[:, fc_cols[1]]

        for s, e in _contact_runs(mask, height, MIN_CONTACT):
            link_contact[s : e + 1, li, 0] = 1.0
            h = float(np.median(height[s : e + 1]))
            if h < TERRAIN_MIN_H:
                continue  # flat floor (or noise) — no edge needed
            if float(np.std(height[s : e + 1])) > HEIGHT_STD_MAX:
                continue  # bobbing height: swing-apex pseudo-contact
            xy = link_pos[s : e + 1, li, :2].mean(axis=0)
            edges.append(ContactEdge(link, "terrain", s, e,
                                     np.array([xy[0], xy[1], h]), height=h))

    off_l = min(sole_offsets["left_foot"])
    off_r = min(sole_offsets["right_foot"])
    if abs(off_l - off_r) > 0.03:
        flags.append("foot_calib_asym")

    # --- pelvis -> terrain (sitting) -------------------------------------
    pelv = jp[:, KIM_PELVIS]
    sit_mask = _stationary(pelv, PELVIS_VEL_MAX, FOOT_ACC_MAX) & (
        pelv[:, 2] < SIT_MAX_Z
    )
    mid_ankle = 0.5 * (jp[:, KIM_LEFT_ANKLE] + jp[:, KIM_RIGHT_ANKLE])
    for s, e in _contact_runs(sit_mask, pelv[:, 2], SIT_MIN_FRAMES):
        # squat vs sit: seated, the feet rest well FORWARD of the pelvis;
        # squatting, they are underneath it (a squat needs no seat)
        feet_off = float(np.linalg.norm(
            (pelv[s : e + 1, :2] - mid_ankle[s : e + 1, :2]).mean(axis=0)
        ))
        if feet_off < SIT_FEET_OFFSET_MIN:
            flags.append("squat_not_sit")
            continue
        seat = float(np.median(pelv[s : e + 1, 2])) - PELVIS_SEAT_DROP
        link_contact[s : e + 1, 4, 0] = 1.0
        if seat < FLOOR_EPS:
            continue  # sitting on the floor
        xy = pelv[s : e + 1, :2].mean(axis=0)
        edges.append(ContactEdge("pelvis", "terrain", s, e,
                                 np.array([xy[0], xy[1], seat]), height=seat))
        flags.append("sit_detected")

    # --- wrists -> object (from the grasp window) -------------------------
    if grasp is not None:
        s, e = grasp.pick_frame, grasp.release_frame
        for link, li in (("left_wrist", 0), ("right_wrist", 1)):
            link_contact[s : e + 1, li, 1] = 1.0
            edges.append(ContactEdge(
                link, "object", s, e, link_pos[s : e + 1, li].mean(axis=0)
            ))

    if any(e.scene == "terrain" for e in edges):
        flags.append("elevated_terrain")
    return InteractionGraph(edges=edges, link_contact=link_contact,
                            link_pos=link_pos, flags=flags)
