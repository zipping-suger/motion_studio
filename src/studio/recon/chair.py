"""Chair reconstruction: a BEHAVE-derived clip of a person lifting,
carrying or sitting on a chair -> a physics trial around the real
BEHAVE mesh (assets/object_mesh/<name>.json, category "chair").

Unlike the Kimodo clips, a BEHAVE clip carries its object: the export
(raw_motion/behave/README.md) stores the chair's canonical-frame pose
per frame next to the retargeted body, so the object is KNOWN and the
question is only WHEN and WHERE the robot touches it. Contact is
therefore detected the way SceneBot (arXiv 2606.27581, Alg. 1) defines
it — a key link is in contact while it sits at the object's surface and
moves with it — and measured against the actual mesh rather than
approximated from hand-pair kinematics as grasp.py must: each hand's
palm centre is expressed in the chair's frame, its surface distance
comes from MuJoCo's own collision distances against the sidecar's
convex hulls, and its object-frame speed says whether it has parked on
the chair or is passing by.

Two levels of contact. A hand's CONTACT window is where the palm stays
at the surface; a hand only counts as holding when the chair moves
meanwhile (an arm hanging beside the backrest is not a grip). Inside
that window the GRIP is the longest stretch where the hand moves
RIGIDLY with the chair (chair-frame speed under RIGID_SPEED): the
BEHAVE hands slide 20-50 cm over the wood between first touch and the
carry, so first contact is the wrong place to measure a grasp. The
pelvis over the seat, still, on a chair that stays put, is a SIT.

The chair is spawned from the contact, not from frame 0: it stands
where the first grip begins (floored and uprighted, nudged out of the
robot's approach sweep), keeps that pose until then, follows the clip's
own object trajectory while held — pushed out of any body geom it
passes through — and is parked after the last release. The wrists are
re-aimed by wrist-only IK to WRAP the chair member under each palm
(the member's axis read off the nearest hull), the fingers close on a
member-sized bar and conform onto the hulls, and each palm's grasp
pocket over its grip, in the chair's frame, is the anchor the solve's
grip reward holds. The trial declares those grips, the body supports
and the hull object (recon.spec / solve.spec); `TASK` is the ReconTask
(recon.run) the chair task runs through.
"""

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from . import assets, layout, mjcf
from .graph import KEY_LINKS, ContactEdge, build_interaction_graph
from .grasp import GraspInfo, hand_tracks
from .loader import KIM_PELVIS
from .objects import (GRASPS, OBJECT_MESH_DIR, chair_objects, infer_by_name,
                      load_sidecar)
from .pick import (HAND_NOISE_SCALE, HOLD_PAD_S, IK_ORI_FLAG,
                   POWER_CLOSE_FRACTION, SIDE_BLEND_S, WRAP_SQUEEZE,
                   _closure_profile, calibrate_grasp, palm_grip_track)
from .pole import (HELD_START_SLACK, _close_hand, _conform_fingers,
                   _parse_hand_window, _runs, pole_grasp_frame)
from .scene import HAND_ENGAGE_FRAMES
from .signal import smooth3
from .robot import robot_model
from .spec import (SPAWN_SKIP_PEN, Built, Grip, Interaction, ObjectSpec,
                   TrialSpec, assemble_qpos)

# --- detection ---------------------------------------------------------
# Surface distance of the palm centre (the Kimodo hand-tip keypoint, at
# the wrist_yaw frame's +0.10 m) to the chair. A gripping BrainCo palm
# sits 3-7 cm off the wood on the BEHAVE lifts, up to ~12 cm when the
# seat is held by the fingertips from below; the free hand stays 13 cm
# and further away, and swings.
CONTACT_ENTER = 0.10   # m: to engage, the palm must come this close...
PARK_SPEED = 0.40      # m/s: ...and be this quiet in the CHAIR's frame
CONTACT_EXIT = 0.16    # m: an engaged palm lets go past this...
EXIT_FRAMES = 6        # ...for this many consecutive frames
PARK_R = 0.06          # m: chair-frame drift within which the engaging
                       # hand already sat on the chair (walk-back)
PARK_LOOKBACK = 45     # frames searched back from the engage moment
BRIDGE = 10            # frames: bridge brief contact dropouts (regrips)
MIN_HOLD = 15          # frames: shortest credible contact
# the grip: the hand moves WITH the chair. 9-frame smoothing rides over
# the BEHAVE object track's jitter; at 0.20 m/s every carry keeps one
# long segment whose chair-frame hand scatter is 4-8 cm, while the
# first-touch slide (20-50 cm) falls out
RIGID_SPEED = 0.20     # m/s: chair-frame palm speed of a gripping hand
RIGID_SMOOTH = 9       # frames
RIGID_BRIDGE = 10      # frames
# the chair's own motion: a hold has to overlap it
OBJ_MOVE_SPEED = 0.05  # m/s
OBJ_MOVE_RATE = 0.20   # rad/s
OBJ_MOVE_BRIDGE = 15   # frames
OBJ_MOVE_MIN = 5       # frames
OBJ_MOVE_TRAVEL = 0.08  # m: a motion run must carry the chair this far...
OBJ_MOVE_ROT = 15.0    # deg: ...or turn it this much; a simulated
                       # chair jitters faster than OBJ_MOVE_SPEED
HOLD_OVERLAP = MIN_HOLD  # frames a hold must share with the motion
NUDGE_MIN = 0.03       # m: object travel before the grab worth a flag
# sitting: the pelvis keypoint parks over the seat, 13-18 cm above it
SIT_XY_MARGIN = 0.06   # m: footprint slack around the seat half-extents
SIT_Z_MAX = 0.30       # m: pelvis height band above the seat surface
SIT_SPEED = 0.15       # m/s: chair-frame pelvis speed of a seated body
SIT_MIN = 15           # frames
SIT_OBJ_TRAVEL = 0.08  # m: a sat-on chair may shift this much

# --- object placement --------------------------------------------------
REST_Z_MAX = 0.03      # m: a resting chair's base sits within this of
REST_TILT_MAX = 5.0    # deg: ...the floor, this close to upright
POS_BLEND = 12         # frames: decay of the rest->clip pose offset
PARK_BLEND_S = 0.6     # s: post-release settle onto the floor
PROBE_R = 0.001        # m: the point-probe sphere
# the resting chair is nudged so the robot's approach never sweeps
# through it (the box's placement search, xy only: the pose is the
# clip's, only the retarget's residual is corrected)
REST_SEARCH = 0.08     # m: half-range of the xy search
REST_STEP = 0.02       # m
REST_TOL = 0.005       # m: approach overlap tolerated without a shift

# --- grasp ---------------------------------------------------------------
# a chair member (seat edge, back rail, leg) is ~3 cm thick: the finger
# closure is calibrated on that bar and then conformed onto the hulls
BAR_DIAMETER = 0.03
MEMBER_ELONGATION = 1.3     # a hull this much longer than wide is a bar
                            # whose axis the fingers wrap; a slab's
                            # wrap axis is its edge under the palm
BODY_CLEAR_MARGIN = 0.004   # m: gap a held chair keeps to palms and body
BODY_CLEAR_SMOOTH = 9       # frames of smoothing on the clearing offset
BODY_SHIFT_MAX = 0.06       # m: larger per-frame shifts are capped and
                            # flagged — the retarget is off, not the grip
# the arm re-pose: pocket onto the member centre line, palm wrapping
# it where that fits. Seven joints, damped, pulled back toward the
# clip's arm in the nullspace so a few centimetres of reach do not
# contort the arm; candidates that put the hand or forearm through the
# chair lose to ones that do not
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw")
ARM_ITERS = 60
ARM_DAMPING = 0.05
ARM_STEP_MAX = 0.25         # rad per joint per iteration
ARM_ROT_WEIGHT = 0.12       # m per rad: orientation error's weight next
                            # to position in the damped solve
ARM_NULL_GAIN = 0.2         # pull toward the clip's joints per iteration
ARM_POS_TOL = 0.03          # m: the pocket lands within this of the anchor
                            # (the fingers close the rest)
ARM_REACH_MAX = 0.18        # m: farther anchors are out of the arm's
                            # correction budget -> keep the clip's arm
ARM_PEN_TOL = 0.010         # m: hand/forearm overlap with the chair a
                            # candidate may keep (the conform pass and
                            # the physics absorb this much)
# the grasp fit: where along the member, and at which palm orientation,
# the closed BrainCo hand fits the chair without cutting into the
# neighbouring members (the human's hand was thinner and fitted where
# the G1's does not). Searched once in the chair frame, so the chair's
# motion never enters
FIT_SLIDE = 0.10            # m: anchor slid this far each way along the
FIT_SLIDE_STEP = 0.02       # member axis...
FIT_YAW_STEP = 30.0         # deg: ...at wrap yaws this far apart
FIT_KEEP = 6                # best fits handed to the arm IK, in order
FIT_SLIDE_COST = 5.0        # rad per m: a slide along the member costs
                            # this much turn away from the clip's palm
                            # (10 cm ~ 30 deg: the same grasp a hand's
                            # width over beats a different grasp)
ARM_DEPARTURE_MAX = 1.5     # rad (7-joint norm): a re-posed arm farther
                            # from the clip than this is a contortion,
                            # not a reach
# how a hand holds the chair: WRAP a member the closed hand can enclose
# (legs, rails: ~5 cm across), or PRESS the palm flat against a face it
# cannot (seat underside, backrest slats, the seat's thick edge). The
# press has fewer constraints (position + palm normal, yaw free) and is
# also the fallback when no wrap fits or reaches
WRAP_MAX_WIDTH = 0.07       # m: both cross extents of a wrap-able member
PRESS_STANDOFF = 0.004      # m: palm face to the pressed surface
PRESS_SLIDE = 0.06          # m: press point slid this far over the face...
PRESS_SLIDE_STEP = 0.03     # ...in the surface plane
# body supports: a carried chair resting on the torso, hip, thighs or
# forearms is a contact the solve should keep, not just tolerate
SUPPORT_DIST = 0.015        # m: body geom within this of the hulls
SUPPORT_MIN = 15            # frames
SUPPORT_BRIDGE = 10         # frames
SLIVER_MAX = 0.03           # m: a hull no larger than this is a CoACD
                            # sliver, never the member under a hand

OBJECT_KEYWORDS = (("chair", "chairwood"),)

# the body collision geoms a carried chair meets (mjcf restores them)
BODY_COLLISION_GEOMS = mjcf.BODY_COLLISION_RESTORE


def available_objects() -> Tuple[str, ...]:
    """The chair objects whose measured sidecar exists (category
    "chair"; the pole sidecars sharing the dir are skipped)."""
    return chair_objects()


CHAIR_OBJECTS = available_objects()


# ------------------------------------------------------------- object --

@dataclass
class ChairSpec:
    """One measured chair (see behave_pipeline's export_to_studio.py):
    canonical frame +z up, leg tips on z=0, footprint centred, +x
    facing."""
    name: str
    obj_file: Path              # visual mesh, referenced in place
    mesh_scale: float           # raw -> canonical affine, split for
    mesh_quat: np.ndarray       # MuJoCo: asset scale + geom pos/quat
    mesh_pos: np.ndarray
    height: float
    seat_z: float               # seat surface height
    seat_center: np.ndarray     # (2,) seat footprint centre
    seat_half: np.ndarray       # (2,) seat footprint half-extents
    base: Dict                  # footprint box
    mass: float
    com: np.ndarray
    inertia_per_mass: np.ndarray
    hulls: List[Path]           # convex collision pieces (same affine)
    floor_z: float = 0.0        # body-origin height with the hulls just
                                # touching the floor: the convex pieces
                                # bulge below the visual mesh's leg tips


def infer_object(clip_name: str, object_param: str,
                 clip_object: Optional[str] = None) -> str:
    """The chair to reconstruct: an explicit choice, else the object
    the clip names (BEHAVE npz `object_name`), else a clip-name
    keyword, else the first measured chair."""
    if object_param != "auto":
        if object_param not in CHAIR_OBJECTS:
            raise SystemExit(f"unknown chair object {object_param!r} "
                             f"(use auto, {', '.join(CHAIR_OBJECTS)})")
        return object_param
    if not CHAIR_OBJECTS:
        raise SystemExit(f"no chair sidecar under {OBJECT_MESH_DIR} "
                         "(behave_pipeline/scripts/export_to_studio.py)")
    if clip_object and clip_object in CHAIR_OBJECTS:
        return clip_object
    name = clip_name.lower()
    for obj in CHAIR_OBJECTS:
        if obj in name:
            return obj
    return infer_by_name(clip_name, OBJECT_KEYWORDS, CHAIR_OBJECTS,
                         CHAIR_OBJECTS[0])


def _obj_vertices(path: Path) -> np.ndarray:
    return np.array([[float(x) for x in line.split()[1:4]]
                     for line in path.read_text().splitlines()
                     if line.startswith("v ")])


def load_spec(name: str) -> ChairSpec:
    d = load_sidecar(name, "chair-object")
    if d.get("category") != "chair":
        raise SystemExit(f"{OBJECT_MESH_DIR / (name + '.json')} "
                         "is not a chair sidecar")
    hulls = [OBJECT_MESH_DIR / h for h in d["collision_hulls"]]
    R = _quat_mat(np.array(d["mesh_quat"], dtype=float))
    z_min = min(float((_obj_vertices(h) @ R.T * float(d["mesh_scale"])
                       + np.array(d["mesh_pos"]))[:, 2].min())
                for h in hulls)
    return ChairSpec(
        name=d["name"],
        obj_file=OBJECT_MESH_DIR / d["obj_file"],
        mesh_scale=float(d["mesh_scale"]),
        mesh_quat=np.array(d["mesh_quat"], dtype=float),
        mesh_pos=np.array(d["mesh_pos"], dtype=float),
        height=float(d["height"]),
        seat_z=float(d["seat_z"]),
        seat_center=np.array(d["seat_center"], dtype=float),
        seat_half=np.array(d["seat_half"], dtype=float),
        base=d["base"],
        mass=float(d["mass"]),
        com=np.array(d["com"], dtype=float),
        inertia_per_mass=np.array(d["inertia_per_mass"], dtype=float),
        hulls=hulls,
        floor_z=max(-z_min, 0.0),
    )


def clip_object_pose(meta: Dict) -> np.ndarray:
    """The clip's own object trajectory (T, 7: pos + wxyz quat of the
    canonical frame), as the loader passes it through."""
    obj = meta.get("object_pose")
    if obj is None:
        raise SystemExit(
            f"{meta.get('file_path')}: no object trajectory in the clip "
            "(object_pos / object_quat_wxyz); the chair recon places the "
            "chair from the clip's object pose")
    return np.asarray(obj, dtype=float)


# ---------------------------------------------------- frame helpers --

def _quat_mat(q: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def _quat_mats(quats: np.ndarray) -> np.ndarray:
    return np.stack([_quat_mat(q) for q in quats], axis=0)


def to_object_frame(track: np.ndarray, obj: np.ndarray) -> np.ndarray:
    """World-frame points (T, 3) -> the chair's canonical frame."""
    Rs = _quat_mats(obj[:, 3:])
    return np.einsum("tji,tj->ti", Rs, track - obj[:, :3])


def _tilt_deg(q: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(_quat_mat(q)[2, 2], -1, 1))))


def _floored(pose: np.ndarray, floor_z: float = 0.0) -> np.ndarray:
    """The same chair standing on the floor: yaw kept, tilt dropped,
    the hulls just touching the ground."""
    x = _quat_mat(pose[3:])[:, 0]
    yaw = float(np.arctan2(x[1], x[0])) if np.hypot(x[0], x[1]) > 1e-6 \
        else 0.0
    out = np.array(pose, dtype=float)
    out[2] = floor_z
    out[3:] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    return out


def _quat_pow(q: np.ndarray, w: float) -> np.ndarray:
    """q ** w for a unit quaternion (fraction of its rotation)."""
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, np.asarray(q, dtype=float), 1.0)
    out = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mju_quatIntegrate(out, vel, float(w))
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=float),
                       np.asarray(b, dtype=float))
    return out


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _blend_pose(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """Pose a -> b at fraction w (position lerp, rotation slerp)."""
    out = np.zeros(7)
    out[:3] = (1 - w) * a[:3] + w * b[:3]
    out[3:] = _quat_mul(_quat_pow(_quat_mul(b[3:], _quat_conj(a[3:])), w),
                        a[3:])
    return out


# ----------------------------------------------------------- probe --

class ChairProbe:
    """Surface distances to the chair, in its canonical frame: the
    sidecar's convex hulls as static geoms plus a point probe, measured
    with MuJoCo's own collision distance (negative inside)."""

    def __init__(self, spec: ChairSpec):
        sc = spec.mesh_scale
        p, q = spec.mesh_pos, spec.mesh_quat
        meshes = "\n".join(
            f'<mesh name="h{i}" file="{h}" scale="{sc:.6f} {sc:.6f} {sc:.6f}"/>'
            for i, h in enumerate(spec.hulls))
        geoms = "\n".join(
            f'<geom name="h{i}" type="mesh" mesh="h{i}" '
            f'pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
            f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" '
            'contype="0" conaffinity="0"/>'
            for i in range(len(spec.hulls)))
        xml = (f"<mujoco><asset>{meshes}</asset><worldbody>"
               f'<body name="chair">{geoms}</body>'
               '<body name="probe" mocap="true"><geom name="probe" '
               f'type="sphere" size="{PROBE_R}" contype="0" conaffinity="0"/>'
               "</body></worldbody></mujoco>")
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.hulls = [self.model.geom(f"h{i}").id
                      for i in range(len(spec.hulls))]
        self.probe = self.model.geom("probe").id
        self._fromto = np.zeros(6)
        # CoACD slivers: too small to be a member, skipped by closest()
        self.sliver = [bool(np.ptp(self.hull_vertices(i), axis=0).max()
                            <= SLIVER_MAX) for i in range(len(spec.hulls))]

    def closest(self, p_local: np.ndarray, distmax: float = 1.0
                ) -> Tuple[float, int, np.ndarray]:
        """(surface distance, index of the nearest hull, the closest
        point on it)."""
        self.data.mocap_pos[0] = p_local
        mujoco.mj_kinematics(self.model, self.data)
        best, which, point = distmax, -1, np.array(p_local, dtype=float)
        for i, g in enumerate(self.hulls):
            if self.sliver[i]:
                continue
            d = mujoco.mj_geomDistance(self.model, self.data, self.probe, g,
                                       best, self._fromto)
            if d < best:
                best, which, point = d, i, self._fromto[3:].copy()
        return float(best) + PROBE_R, which, point

    def nearest(self, p_local: np.ndarray,
                distmax: float = 1.0) -> Tuple[float, int]:
        """(surface distance, index of the nearest hull)."""
        return self.closest(p_local, distmax)[:2]

    def distance(self, p_local: np.ndarray, distmax: float = 1.0) -> float:
        return self.nearest(p_local, distmax)[0]

    def distances(self, pts: np.ndarray) -> np.ndarray:
        return np.array([self.distance(p) for p in pts])

    def hull_vertices(self, i: int) -> np.ndarray:
        """Vertices of hull i in the canonical frame."""
        m, g = self.model, self.hulls[i]
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]
                        + m.mesh_vertnum[mid]]
        return v @ _quat_mat(m.geom_quat[g]).T + m.geom_pos[g]


# ---------------------------------------------------------- detection --

@dataclass
class ChairInfo:
    kind: str                   # "carry" | "sit" | "none"
    hands: List[str]            # gripping hands, first-to-grip first
    hand_grabs: List[int]       # per-hand GRIP start, aligned with hands
    hand_ends: List[int]        # per-hand grip end (inclusive)
    grab_frame: int             # the first grip's start
    end_frame: int              # the last grip's end
    starts_held: bool
    motion: Optional[Tuple[int, int]]   # the chair's own motion span
    sit: Optional[Tuple[int, int]]      # pelvis-on-seat window
    contacts: Dict[str, Tuple[int, int]] = field(default_factory=dict)
                                # each gripping hand's full contact
                                # window (the grip lies inside it)
    touches: Dict[str, Tuple[int, int]] = field(default_factory=dict)
                                # hand contacts that are not holds
    transport: float = 0.0      # peak chair xy travel over the hold
    lift: float = 0.0           # peak chair rise over the hold
    quality_flags: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.kind != "none"

    @property
    def ends(self) -> List[int]:
        return self.hand_ends


def _park_frame(loc: np.ndarray, dist: np.ndarray, s: int) -> int:
    """Earliest frame from which the hand stays within PARK_R of its
    chair-frame position at s (and within reach): when it actually
    settled on the chair."""
    ref = loc[s]
    park = s
    for t in range(s, max(0, s - PARK_LOOKBACK) - 1, -1):
        if (np.linalg.norm(loc[t] - ref) > PARK_R
                or dist[t] > CONTACT_EXIT):
            break
        park = t
    return park


def _contact_runs(dist: np.ndarray, loc: np.ndarray,
                  fps: float) -> List[Tuple[int, int]]:
    """Contact intervals of one hand: engaged when near AND parked in
    the chair frame, released after EXIT_FRAMES beyond CONTACT_EXIT."""
    spd = np.linalg.norm(np.gradient(smooth3(loc), axis=0), axis=1) * fps
    T = len(dist)
    runs: List[Tuple[int, int]] = []
    t = 0
    while t < T:
        if dist[t] < CONTACT_ENTER and spd[t] < PARK_SPEED:
            e, out, u = t, 0, t
            while u < T:
                if dist[u] > CONTACT_EXIT:
                    out += 1
                    if out >= EXIT_FRAMES:
                        break
                else:
                    out, e = 0, u
                u += 1
            runs.append((_park_frame(loc, dist, t), e))
            t = e + 1
        else:
            t += 1
    merged: List[Tuple[int, int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= BRIDGE:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def _rigid_runs(loc: np.ndarray, fps: float,
                window: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Stretches inside a contact window where the hand moves WITH the
    chair — its chair-frame speed under RIGID_SPEED."""
    s, e = window
    spd = np.linalg.norm(np.gradient(smooth3(loc, RIGID_SMOOTH), axis=0),
                         axis=1) * fps
    mask = np.zeros(len(loc), dtype=bool)
    mask[s:e + 1] = spd[s:e + 1] < RIGID_SPEED
    return [r for r in _runs(mask, RIGID_BRIDGE)
            if r[1] - r[0] + 1 >= MIN_HOLD]


def object_motion_runs(obj: np.ndarray, fps: float) -> List[Tuple[int, int]]:
    """Frames where the chair translates or turns, as bridged runs."""
    T = len(obj)
    spd = np.linalg.norm(np.gradient(smooth3(obj[:, :3]), axis=0),
                         axis=1) * fps
    rate = np.zeros(T)
    v = np.zeros(3)
    for t in range(1, T):
        mujoco.mju_subQuat(v, obj[t, 3:], obj[t - 1, 3:])
        rate[t] = np.linalg.norm(v) * fps
    rate = np.convolve(rate, np.ones(5) / 5, mode="same")
    mask = (spd > OBJ_MOVE_SPEED) | (rate > OBJ_MOVE_RATE)
    out = []
    for s, e in _runs(mask, OBJ_MOVE_BRIDGE):
        if e - s + 1 < OBJ_MOVE_MIN:
            continue
        travel = np.linalg.norm(obj[s:e + 1, :3] - obj[s, :3], axis=1).max()
        turn = 0.0
        for t in range(s + 1, e + 1):
            mujoco.mju_subQuat(v, obj[t, 3:], obj[s, 3:])
            turn = max(turn, np.linalg.norm(v))
        if travel >= OBJ_MOVE_TRAVEL or np.degrees(turn) >= OBJ_MOVE_ROT:
            out.append((s, e))
    return out


def _overlaps(r: Tuple[int, int], runs: List[Tuple[int, int]]) -> bool:
    return any(min(r[1], me) - max(r[0], ms) + 1 >= HOLD_OVERLAP
               for ms, me in runs)


def _travel(obj: np.ndarray, s: int, e: int) -> Tuple[float, float]:
    """(peak xy travel, peak rise) of the chair over [s, e]."""
    seg = obj[s:e + 1, :3]
    return (float(np.linalg.norm(seg[:, :2] - seg[0, :2], axis=1).max()),
            float((seg[:, 2] - seg[0, 2]).max()))


def _sit_window(meta: Dict, obj: np.ndarray, spec: ChairSpec,
                fps: float) -> Optional[Tuple[int, int]]:
    """Longest window with the pelvis parked over the seat while the
    chair stays put (a sat-on chair may shift a little)."""
    pv = meta["joint_positions"][:, KIM_PELVIS]
    loc = to_object_frame(pv, obj)
    spd = np.linalg.norm(np.gradient(smooth3(loc), axis=0), axis=1) * fps
    cx, cy = spec.seat_center
    hx, hy = spec.seat_half + SIT_XY_MARGIN
    over = ((np.abs(loc[:, 0] - cx) < hx) & (np.abs(loc[:, 1] - cy) < hy)
            & (loc[:, 2] > spec.seat_z)
            & (loc[:, 2] < spec.seat_z + SIT_Z_MAX))
    mask = over & (spd < SIT_SPEED)
    runs = [r for r in _runs(mask, BRIDGE)
            if r[1] - r[0] + 1 >= SIT_MIN
            and _travel(obj, *r)[0] < SIT_OBJ_TRAVEL]
    return max(runs, key=lambda r: r[1] - r[0]) if runs else None


def detect_chair_contact(meta: Dict, qpos: np.ndarray, spec: ChairSpec,
                         obj: Optional[np.ndarray] = None) -> ChairInfo:
    """When each hand holds the chair, where inside that it really
    grips, whether the body sits on it, and the chair's own motion —
    all measured against the clip's object pose and the chair's hulls,
    no knobs.

    A hand's contact run is a hold only if it shares HOLD_OVERLAP
    frames with the chair's motion, and its grip is the longest rigid
    stretch of that hold that also overlaps the motion (a hand resting
    on a chair that has not moved yet is rigid trivially). Any other
    contact is a touch — labelled in a sit trial (hands pushing off
    the seat), merely flagged in a carry. Hands are ordered
    first-to-grip, ties to the longer grip."""
    obj = clip_object_pose(meta) if obj is None else np.asarray(obj)
    fps = meta["fps"]
    T = len(obj)
    lh, rh, _ = hand_tracks(meta)
    probe = ChairProbe(spec)
    flags: List[str] = []

    motion_runs = object_motion_runs(obj, fps)
    motion = ((motion_runs[0][0], motion_runs[-1][1]) if motion_runs
              else None)

    grips: Dict[str, Tuple[int, int]] = {}
    contacts: Dict[str, Tuple[int, int]] = {}
    touches: Dict[str, Tuple[int, int]] = {}
    for hand, tr in (("left", lh), ("right", rh)):
        loc = to_object_frame(tr, obj)
        dist = probe.distances(loc)
        runs = [r for r in _contact_runs(dist, loc, fps)
                if r[1] - r[0] + 1 >= MIN_HOLD]
        if not runs:
            continue
        on = [r for r in runs if _overlaps(r, motion_runs)]
        if not on:
            touches[hand] = max(runs, key=lambda r: r[1] - r[0])
            if motion_runs:
                flags.append(f"touch_dropped_{hand}")
            continue
        hold = max(on, key=lambda r: r[1] - r[0])
        if len(on) > 1:
            flags.append(f"regrip_{hand}")
        rigid = [r for r in _rigid_runs(loc, fps, hold)
                 if _overlaps(r, motion_runs)]
        if not rigid:
            touches[hand] = hold
            flags.append(f"no_rigid_grip_{hand}")
            continue
        grips[hand] = max(rigid, key=lambda r: r[1] - r[0])
        contacts[hand] = hold
        if len(rigid) > 1:
            flags.append(f"slides_{hand}")

    sit = _sit_window(meta, obj, spec, fps)

    if grips:
        kind = "carry"
        hands = sorted(grips, key=lambda h: (grips[h][0],
                                             grips[h][0] - grips[h][1]))
        grabs = [grips[h][0] for h in hands]
        ends = [grips[h][1] for h in hands]
        s, e = grabs[0], max(ends)
        starts_held = s <= HELD_START_SLACK
        if starts_held:
            grabs = [0 if g <= HELD_START_SLACK else g for g in grabs]
            contacts = {h: (0 if c[0] <= HELD_START_SLACK else c[0], c[1])
                        for h, c in contacts.items()}
            s = 0
            flags.append("starts_held")
        if len(hands) == 1:
            flags.append("one_hand_hold")
        if e < T - 6:
            flags.append("released_early")
        if s > 0 and np.linalg.norm(obj[:s, :2] - obj[0, :2],
                                    axis=1).max() > NUDGE_MIN:
            flags.append("object_nudged")   # moved before any hand held it
        transport, lift = _travel(obj, s, e)
    elif sit is not None and _travel(obj, 0, T - 1)[0] < SIT_OBJ_TRAVEL:
        kind, hands, grabs, ends = "sit", [], [], []
        s, e = sit
        starts_held = False
        transport = lift = 0.0
        flags.append("sit")
    else:
        kind, hands, grabs, ends = "none", [], [], []
        s, e, starts_held = 0, T - 1, False
        transport = lift = 0.0
        flags.append("object_moves_unheld" if motion_runs
                     else "no_interaction")
    if sit is not None and kind == "carry":
        flags.append("sit")
    return ChairInfo(kind, hands, [int(g) for g in grabs],
                     [int(x) for x in ends], int(s), int(e), starts_held,
                     motion, sit, contacts, touches, transport, lift, flags)


def apply_hand_windows(info: ChairInfo, meta: Dict, qpos: np.ndarray,
                       left: str = "auto",
                       right: str = "auto") -> ChairInfo:
    """User-specified per-hand GRIP windows over the detection: each
    hand is auto (keep the detection), off (never holds), or an
    explicit inclusive window S-E (see pole.apply_hand_windows). Any
    override rebuilds the ChairInfo from the resulting windows; a forced
    window is its own contact window."""
    T = len(qpos)
    specs = {"left": _parse_hand_window(left, "left", T),
             "right": _parse_hand_window(right, "right", T)}
    if all(v is None for v in specs.values()):
        return info
    detected = dict(zip(info.hands, zip(info.hand_grabs, info.hand_ends)))
    windows: Dict[str, Tuple[int, int]] = {}
    contacts: Dict[str, Tuple[int, int]] = {}
    for hand, spec in specs.items():
        if spec is None:
            if hand in detected:
                windows[hand] = detected[hand]
                contacts[hand] = info.contacts.get(hand, detected[hand])
        elif spec != ():
            windows[hand] = spec
            contacts[hand] = spec
    obj = clip_object_pose(meta)
    flags = ["hand_windows"]
    if not windows:
        if info.sit is not None and info.motion is None:
            return dataclasses.replace(
                info, kind="sit", hands=[], hand_grabs=[], hand_ends=[],
                grab_frame=info.sit[0], end_frame=info.sit[1],
                starts_held=False, contacts={}, transport=0.0, lift=0.0,
                quality_flags=flags + ["sit"])
        return dataclasses.replace(
            info, kind="none", hands=[], hand_grabs=[], hand_ends=[],
            grab_frame=0, end_frame=T - 1, starts_held=False, contacts={},
            quality_flags=flags + ["no_interaction"])
    hands = sorted(windows, key=lambda h: (windows[h][0],
                                           windows[h][0] - windows[h][1]))
    grabs = [windows[h][0] for h in hands]
    ends = [windows[h][1] for h in hands]
    s, e = grabs[0], max(ends)
    starts_held = s <= HELD_START_SLACK
    if starts_held:
        grabs = [0 if g <= HELD_START_SLACK else g for g in grabs]
        contacts = {h: (0 if c[0] <= HELD_START_SLACK else c[0], c[1])
                    for h, c in contacts.items()}
        s = 0
        flags.append("starts_held")
    if len(hands) == 1:
        flags.append("one_hand_hold")
    if e < T - 6:
        flags.append("released_early")
    if info.sit is not None:
        flags.append("sit")
    transport, lift = _travel(obj, s, e)
    return ChairInfo("carry", hands, [int(g) for g in grabs],
                     [int(x) for x in ends], int(s), int(e), starts_held,
                     info.motion, info.sit, contacts, {}, transport, lift,
                     flags)


# --------------------------------------------------- object trajectory --

def build_chair_trajectory(
    obj_ref: np.ndarray, info: ChairInfo, fps: float, floor_z: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(obj_qpos (T, 7), rest_pos, flags): the clip's chair trajectory
    made consistent with the detected contact.

    Before the first CONTACT the chair stands still where that hand
    meets it — the clip's pose there, floored and uprighted (a chair
    nudged before the touch spawns where it is touched, not where the
    clip began). From the contact the clip's own trajectory takes over
    (a touching hand may slide the chair before gripping it), the rest
    offset decaying over POS_BLEND frames. After the last release the
    chair settles onto the floor and stays. A clip that opens with the
    chair already in the hand keeps its pose from frame 0
    (``spawn_in_air``)."""
    T = len(obj_ref)
    flags: List[str] = []
    if info.kind != "carry":
        rest = _floored(obj_ref[0], floor_z)
        drift = float(np.linalg.norm(obj_ref[:, :3] - obj_ref[0, :3],
                                     axis=1).max())
        if drift > NUDGE_MIN:
            flags.append("object_drifts")
        return np.tile(rest, (T, 1)), rest[:3].copy(), flags

    s = min([c[0] for c in info.contacts.values()] or [info.grab_frame])
    e = info.end_frame
    in_air = (obj_ref[s, 2] - floor_z > REST_Z_MAX
              or _tilt_deg(obj_ref[s, 3:]) > REST_TILT_MAX)
    if s <= HELD_START_SLACK and in_air:
        rest = obj_ref[0].copy()
        s = 0
        flags.append("spawn_in_air")
    else:
        rest = _floored(obj_ref[s], floor_z)
        if in_air:
            flags.append("rest_snapped")

    obj = obj_ref.copy()
    obj[:s] = rest
    off_p = rest[:3] - obj_ref[s, :3]
    off_q = _quat_mul(rest[3:], _quat_conj(obj_ref[s, 3:]))
    for k in range(POS_BLEND + 1):
        t = s + k
        if t > e:
            break
        w = 1.0 - k / POS_BLEND
        obj[t, :3] = obj_ref[t, :3] + w * off_p
        obj[t, 3:] = _quat_mul(_quat_pow(off_q, w), obj_ref[t, 3:])

    if e < T - 1:
        park = _floored(obj_ref[e], floor_z)
        if (obj_ref[e, 2] - floor_z > REST_Z_MAX
                or _tilt_deg(obj_ref[e, 3:]) > REST_TILT_MAX):
            flags.append("release_in_air")
        if float(np.linalg.norm(obj_ref[e:, :3] - obj_ref[e, :3],
                                axis=1).max()) > NUDGE_MIN:
            flags.append("moves_after_release")
        blend = max(min(int(PARK_BLEND_S * fps), T - 1 - e), 1)
        for t in range(e + 1, T):
            w = (t - e) / blend
            obj[t] = park if w >= 1.0 else _blend_pose(obj_ref[e], park, w)
    return obj, obj[0, :3].copy(), flags


# ---------------------------------------------------- grasp geometry --

def member_axis(probe: ChairProbe, anchor: np.ndarray,
                approach: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Unit axis (chair frame) the fingers should wrap at `anchor`: the
    long axis of the nearest hull when it is a bar (leg, rail, post);
    for a slab (seat) the edge under the palm — perpendicular to the
    slab's normal and to the palm's approach direction."""
    dist, i = probe.nearest(anchor)
    v = probe.hull_vertices(i)
    c = v - v.mean(axis=0)
    w, vec = np.linalg.eigh(c.T @ c / len(c))
    sv = np.sqrt(np.maximum(w, 0.0))
    elong = float(sv[2] / max(sv[1], 1e-9))
    kind = "bar"
    if elong >= MEMBER_ELONGATION:
        axis = vec[:, 2]
    else:
        kind = "slab"
        axis = np.cross(vec[:, 0], approach)
        if np.linalg.norm(axis) < 1e-6:
            axis = vec[:, 2]
    axis = axis / np.linalg.norm(axis)
    return axis, {"hull": int(i), "kind": kind, "elongation": round(elong, 2),
                  "surface_dist": round(float(dist), 4)}


def surface_anchor(probe: ChairProbe, pocket_local: np.ndarray
                   ) -> Tuple[np.ndarray, Dict]:
    """Where a palm pocket at `pocket_local` (chair frame) should hold
    the chair: the centre line of the member under it — the closest
    hull point, a bar radius inward along the surface normal."""
    dist, i, point = probe.closest(pocket_local)
    n = pocket_local - point
    nn = np.linalg.norm(n)
    n = (np.sign(dist) * n / nn) if nn > 1e-9 else np.array([0.0, 0.0, 1.0])
    return point - (BAR_DIAMETER / 2) * n, {
        "hull": int(i), "pocket_off_surface": round(float(dist), 4)}


def member_kind(probe: ChairProbe, hull: int) -> Tuple[str, np.ndarray]:
    """("bar" | "face", cross extents) of one hull: a bar is a member
    the closed hand can enclose — both cross-section extents under
    WRAP_MAX_WIDTH — anything wider is a face to press against."""
    v = probe.hull_vertices(hull)
    c = v - v.mean(axis=0)
    w, _vec = np.linalg.eigh(c.T @ c / len(c))
    ext = np.sqrt(12.0 * np.maximum(w, 0.0))   # full extents of a box
    cross = ext[:2]
    return ("bar" if cross.max() <= WRAP_MAX_WIDTH else "face"), cross


def palm_face_pocket(hand: str, sp: float) -> Optional[np.ndarray]:
    """The palm face's centre, PRESS_STANDOFF off the box, in the palm
    site frame — the press's pocket. None on the handless template."""
    model = robot_model()
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{hand[0]}h")
    if gid < 0 or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
        return None
    data = mujoco.MjData(model)
    data.qpos[3] = 1.0
    mujoco.mj_kinematics(model, data)
    sid = model.site(f"{hand}_palm").id
    R = data.site_xmat[sid].reshape(3, 3)
    pc = R.T @ (data.geom_xpos[gid] - data.site_xpos[sid])
    pR = R.T @ data.geom_xmat[gid].reshape(3, 3)
    ext_y = float(np.abs(pR[1]) @ model.geom_size[gid])
    return np.array([pc[0], pc[1] + sp * (ext_y + PRESS_STANDOFF), pc[2]])


def press_frame(pocket: np.ndarray, normal: np.ndarray,
                yaw: float) -> np.ndarray:
    """Palm-site orientation pressing a face: palm normal into the face
    (against its outward `normal`), knuckle line in the face at `yaw`."""
    n = normal / np.linalg.norm(normal)
    sp = 1.0 if pocket[1] >= 0 else -1.0
    y = -sp * n
    t1 = np.cross(n, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(t1) < 1e-6:
        t1 = np.cross(n, np.array([1.0, 0.0, 0.0]))
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    z = np.cos(yaw) * t1 + np.sin(yaw) * t2
    return np.stack([np.cross(y, z), y, z], axis=1)


def plan_press_fit(scene_model, qpos_t: np.ndarray, hand: str,
                   pocket: np.ndarray, probe: ChairProbe,
                   point: np.ndarray, normal: np.ndarray) -> List[Dict]:
    """Where the open hand can press the chair's face near `point`
    (chair frame, outward `normal`): candidates slid over the face in
    its plane and turned about the normal, scored like plan_grasp_fit
    (palm/proximal-finger/forearm overlap, then turn + slide)."""
    hulls = _hull_gids(scene_model)
    geoms = _fit_geoms(scene_model, hand)
    data = mujoco.MjData(scene_model)
    fromto = np.zeros(6)
    sid = scene_model.site(f"{hand}_palm").id
    data.qpos[:] = qpos_t
    mujoco.mj_kinematics(scene_model, data)
    R_h = data.site_xmat[sid].reshape(3, 3).copy()
    p_w = data.site_xpos[sid] + R_h @ pocket
    R_clip = _quat_mat(qpos_t[-4:]).T @ R_h
    n0 = normal / np.linalg.norm(normal)
    t1 = np.cross(n0, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(t1) < 1e-6:
        t1 = np.cross(n0, np.array([1.0, 0.0, 0.0]))
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n0, t1)

    out = []
    quat = np.zeros(4)
    steps = np.arange(-PRESS_SLIDE, PRESS_SLIDE + 1e-9, PRESS_SLIDE_STEP)
    for dx in steps:
        for dy in steps:
            p = point + dx * t1 + dy * t2 + 0.02 * n0
            d, _i, surf = probe.closest(p)
            n = p - surf
            if np.linalg.norm(n) < 1e-6:
                continue
            n = np.sign(d) * n / np.linalg.norm(n)
            if n @ n0 < 0.5:                 # the face turned away
                continue
            for yaw in np.radians(np.arange(0.0, 360.0, FIT_YAW_STEP)):
                R_l = press_frame(pocket, n, yaw)
                R_c = R_h @ R_l.T
                mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R_c).ravel())
                data.qpos[:] = qpos_t
                data.qpos[-7:-4] = p_w - R_c @ surf
                data.qpos[-4:] = quat
                mujoco.mj_kinematics(scene_model, data)
                hit = _deepest(scene_model, data, geoms, hulls, 0.0, fromto)
                pen = float(-hit[0]) if hit is not None else 0.0
                turn = float(np.linalg.norm(_rotvec(R_l @ R_clip.T)))
                out.append({"anchor": surf, "R_local": R_l, "normal": n,
                            "name": f"press{int(np.degrees(yaw))}",
                            "penetration": pen,
                            "slide": float(np.hypot(dx, dy)), "turn": turn})
    out.sort(key=lambda c: (c["penetration"] > ARM_PEN_TOL,
                            round(c["penetration"], 3),
                            c["turn"] + FIT_SLIDE_COST * abs(c["slide"])))
    return out


def detect_supports(model, qpos: np.ndarray, s: int, e: int) -> List[Dict]:
    """Body geoms (everything the chair collides with but the hands)
    that rest against the chair over the hold: per geom, runs of at
    least SUPPORT_MIN frames within SUPPORT_DIST of the hulls."""
    hulls = _hull_gids(model)
    # the wrists ride with the hands: their contact is the grasp's
    body = {g: n for g, n in _colliding_geoms(model).items()
            if not _is_hand_geom(n) and "wrist" not in n}
    if not hulls or not body:
        return []
    data = mujoco.MjData(model)
    fromto = np.zeros(6)
    T = len(qpos)
    dist = {g: np.full(T, np.inf) for g in body}
    for t in range(s, e + 1):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        for g in body:
            dist[g][t] = min(mujoco.mj_geomDistance(model, data, g, h,
                                                    SUPPORT_DIST, fromto)
                             for h in hulls)
    out = []
    for g, n in body.items():
        for a, b in _runs(dist[g] < SUPPORT_DIST, SUPPORT_BRIDGE):
            if b - a + 1 < SUPPORT_MIN:
                continue
            out.append({"geom": n, "start": int(a), "end": int(b),
                        "gap_mm": round(float(np.median(
                            np.clip(dist[g][a:b + 1], -1, 1))) * 1000, 1)})
    return out


def _arm_setup(model, hand: str):
    """(palm site id, its body id, arm qpos addrs, dof addrs, joint
    ranges, body-contract columns) for one arm's seven joints."""
    sid = model.site(f"{hand}_palm").id
    jids = [model.joint(f"{hand}_{j}_joint").id for j in ARM_JOINTS]
    qadr = np.array([int(model.jnt_qposadr[j]) for j in jids])
    vadr = np.array([int(model.jnt_dofadr[j]) for j in jids])
    from .loader import DOF_NAMES
    cols = [7 + DOF_NAMES.index(f"{hand}_{j}_joint") for j in ARM_JOINTS]
    return (sid, int(model.site_bodyid[sid]), qadr, vadr,
            model.jnt_range[jids].copy(), cols)


def _rotvec(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def arm_ik_frame(model, data, setup, base: np.ndarray, arm: np.ndarray,
                 arm_ref: np.ndarray, pocket: np.ndarray,
                 target_p: np.ndarray, R_t: np.ndarray):
    """Damped least-squares IK of one arm onto a pocket position and a
    palm orientation, with the nullspace pulled toward `arm_ref`.
    Returns (arm joints, position error, orientation error rad, palm
    site rotation); `data` is left at the returned pose."""
    sid, bid, qadr, vadr, jrange, _cols = setup
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(ARM_ITERS):
        base[qadr] = arm
        data.qpos[:] = base
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        R_c = data.site_xmat[sid].reshape(3, 3)
        p = data.site_xpos[sid] + R_c @ pocket
        e_p = target_p - p
        e_r = _rotvec(R_t @ R_c.T)
        if np.linalg.norm(e_p) < 1e-3 and np.linalg.norm(e_r) < 1e-3:
            break
        mujoco.mj_jac(model, data, jacp, jacr, p, bid)
        J = np.vstack([jacp[:, vadr], ARM_ROT_WEIGHT * jacr[:, vadr]])
        e = np.concatenate([e_p, ARM_ROT_WEIGHT * e_r])
        JJt = J @ J.T + (ARM_DAMPING ** 2) * np.eye(6)
        J_pinv = J.T @ np.linalg.solve(JJt, np.eye(6))
        dq = J_pinv @ e
        dq += (np.eye(len(arm)) - J_pinv @ J) @ (ARM_NULL_GAIN
                                                 * (arm_ref - arm))
        dq = np.clip(dq, -ARM_STEP_MAX, ARM_STEP_MAX)
        arm = np.clip(arm + dq, jrange[:, 0], jrange[:, 1])
    base[qadr] = arm
    data.qpos[:] = base
    mujoco.mj_kinematics(model, data)
    R_c = data.site_xmat[sid].reshape(3, 3)
    p = data.site_xpos[sid] + R_c @ pocket
    return (arm, float(np.linalg.norm(target_p - p)),
            float(np.linalg.norm(_rotvec(R_t @ R_c.T))), R_c.copy())


def _fit_geoms(scene_model, hand: str) -> List[int]:
    """What a grasp pose must keep out of the chair: the palm box, the
    proximal finger capsules and the arm's own collision geoms
    (wrist, elbow, shoulder yaw). The distal capsules are left out —
    they wrap whatever thickness the member has, and the conform pass
    sets them afterwards."""
    out = []
    for g in range(scene_model.ngeom):
        n = mujoco.mj_id2name(scene_model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if (n == f"{hand[0]}h" or re.fullmatch(rf"{hand[0]}h_\w+0", n)
                or re.fullmatch(rf"{hand}_(wrist|elbow_yaw|shoulder_yaw)"
                                r"_collision", n)):
            out.append(g)
    return out


def plan_grasp_fit(scene_model, qpos_t: np.ndarray, hand: str,
                   pocket: np.ndarray, anchor: np.ndarray,
                   axis_local: np.ndarray) -> List[Dict]:
    """Where the closed hand fits the chair near `anchor`: candidate
    (anchor, palm orientation) pairs in the chair frame, best first.

    The clip's hand pose at frame `qpos_t` (fingers already closed) is
    held fixed and the CHAIR is placed instead, so that the candidate
    anchor sits in the pocket with the candidate palm orientation —
    the fit is relative, and the clip's forearm rides along as a proxy
    for the re-posed one. Candidates: the anchor slid along the member
    axis (FIT_SLIDE), the wrap frame at FIT_YAW_STEP yaws about the
    axis in both knuckle senses, plus the clip's own orientation.
    Scored by palm/proximal-finger/forearm overlap with the hulls, then
    by the turn away from the clip's palm plus FIT_SLIDE_COST times the
    slide."""
    hulls = _hull_gids(scene_model)
    geoms = _fit_geoms(scene_model, hand)
    data = mujoco.MjData(scene_model)
    fromto = np.zeros(6)
    sid = scene_model.site(f"{hand}_palm").id
    data.qpos[:] = qpos_t
    mujoco.mj_kinematics(scene_model, data)
    R_h = data.site_xmat[sid].reshape(3, 3).copy()
    p_w = data.site_xpos[sid] + R_h @ pocket
    R_c0 = _quat_mat(qpos_t[-4:])
    R_clip = R_c0.T @ R_h                      # clip palm, chair frame
    a = axis_local / np.linalg.norm(axis_local)

    def frames():
        yield "clip", R_clip
        for z_sign in (1.0, -1.0):
            for yaw in np.radians(np.arange(0.0, 360.0, FIT_YAW_STEP)):
                yield f"wrap{int(np.degrees(yaw))}{'+' if z_sign > 0 else '-'}", \
                    pole_grasp_frame(pocket, a, yaw, z_sign)

    out = []
    quat = np.zeros(4)
    slides = np.arange(-FIT_SLIDE, FIT_SLIDE + 1e-9, FIT_SLIDE_STEP)
    for k in slides:
        anc = anchor + k * a
        for name, R_l in frames():
            R_c = R_h @ R_l.T
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R_c).ravel())
            data.qpos[:] = qpos_t
            data.qpos[-7:-4] = p_w - R_c @ anc
            data.qpos[-4:] = quat
            mujoco.mj_kinematics(scene_model, data)
            hit = _deepest(scene_model, data, geoms, hulls, 0.0, fromto)
            pen = float(-hit[0]) if hit is not None else 0.0
            turn = float(np.linalg.norm(_rotvec(R_l @ R_clip.T)))
            out.append({"anchor": anc, "R_local": R_l, "name": name,
                        "penetration": pen, "slide": float(k), "turn": turn})
    out.sort(key=lambda c: (c["penetration"] > ARM_PEN_TOL,
                            round(c["penetration"], 3),
                            c["turn"] + FIT_SLIDE_COST * abs(c["slide"])))
    return out


def retarget_chair_hand(
    qpos_robot: np.ndarray, hand: str, pocket: np.ndarray,
    obj_qpos: np.ndarray, anchor: np.ndarray, axis_local: np.ndarray,
    grab: int, end: int, fps: float, scene_model=None,
    fits: Optional[List[Dict]] = None,
    finger_pose: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """Re-pose one gripping ARM so its palm pocket sits on the member
    centre line and the fingers wrap the member — the pole's wrist
    re-aim was enough for hands the clip already had on the object;
    the BEHAVE retargets leave the G1's pocket 2-10 cm off the wood,
    and the body clearing can move it farther, so the reach has to come
    from the arm.

    From SIDE_BLEND_S before the grip the arm blends out of the clip's
    pose into the solve and holds it through the grip; an early
    release blends back. Per frame, damped least-squares over the seven
    arm joints onto the anchor (carried by the chair) and the wrap frame
    (pole.pole_grasp_frame on the member axis), warm-started, pulled
    toward the clip's own arm in the nullspace. Candidates are the
    planned grasp fits (`fits`, plan_grasp_fit: anchor slid along the
    member, wrap orientations that keep the hand out of the chair),
    each carried by the chair, plus the clip's own palm orientation
    with only the position corrected. Scored on a frame stride:
    feasible when the pocket lands within ARM_POS_TOL and the
    orientation within IK_ORI_FLAG of its target; then, given
    `scene_model`, by whether the hand and forearm stay out of the
    chair (ARM_PEN_TOL); then by the smallest joint-space departure
    from the clip. Returns (qpos, info, the anchor used). Falls back to
    the clip's own arm (mode "reference") when nothing is feasible, or
    when the anchor sits farther than ARM_REACH_MAX from the clip's
    pocket."""
    model = robot_model()
    data = mujoco.MjData(model)
    setup = _arm_setup(model, hand)
    sid, _bid, qadr, _vadr, _jrange, cols = setup
    T = len(qpos_robot)
    s, e = grab, end
    Rs = _quat_mats(obj_qpos[:, 3:])
    anchor_t = obj_qpos[:, :3] + np.einsum("tij,j->ti", Rs, anchor)

    bases = layout.to_model(model, qpos_robot)
    data.qpos[:] = bases[s]
    mujoco.mj_kinematics(model, data)
    ref_dists = []
    for t in range(s, e + 1, 3):
        data.qpos[:] = bases[t]
        mujoco.mj_kinematics(model, data)
        pw = data.site_xpos[sid] + data.site_xmat[sid].reshape(3, 3) @ pocket
        ref_dists.append(np.linalg.norm(pw - anchor_t[t]))
    ref_dist = float(np.mean(ref_dists)) if ref_dists else 0.0
    if ref_dist > ARM_REACH_MAX:
        return qpos_robot, {"hand": hand, "mode": "reference",
                            "fallback": "out_of_reach",
                            "pocket_anchor_ref": round(ref_dist, 3)}, anchor

    blend = max(int(SIDE_BLEND_S * fps), 1)
    t0 = max(s - blend, 0)
    w = np.zeros(T)
    w[t0:s + 1] = np.linspace(0.0, 1.0, s - t0 + 1)
    w[s:e + 1] = 1.0
    if e < T - 6:
        out_end = min(e + blend, T - 1)
        w[e:out_end + 1] = np.linspace(1.0, 0.0, out_end - e + 1)

    # the clip's own palm orientation per frame: the position-only
    # candidate keeps it and just brings the pocket to the member
    R_clip = np.zeros((T, 3, 3))
    for t in range(t0, min(e + blend, T - 1) + 1):
        data.qpos[:] = bases[t]
        mujoco.mj_kinematics(model, data)
        R_clip[t] = data.site_xmat[sid].reshape(3, 3)

    # candidates: the planned fits (chair-frame pose, carried by the
    # chair) and the clip's own orientation at the clip's anchor
    cands = [{"anchor": f["anchor"], "R_local": f["R_local"],
              "name": f["name"]} for f in (fits or [])[:FIT_KEEP]]
    cands.append({"anchor": anchor, "R_local": None, "name": "clip"})
    for c in cands:
        c["anchor_t"] = obj_qpos[:, :3] + np.einsum("tij,j->ti", Rs,
                                                    c["anchor"])

    def frame_at(cand, t: int) -> np.ndarray:
        if cand["R_local"] is None:
            return R_clip[t]
        return Rs[t] @ cand["R_local"]

    if scene_model is not None:
        hulls = _hull_gids(scene_model)
        probe_geoms = _fit_geoms(scene_model, hand)
        sdata = mujoco.MjData(scene_model)
        fromto = np.zeros(6)

    def penetration(t: int, arm: np.ndarray) -> float:
        """Deepest overlap of the palm, proximal fingers and forearm
        with the chair at frame t under `arm`, fingers closed as they
        will be (0 without a scene)."""
        if scene_model is None:
            return 0.0
        q = np.concatenate([bases[t], obj_qpos[t]])
        q[qadr] = arm
        for adr, angle in (finger_pose or {}).items():
            q[adr] = angle
        sdata.qpos[:] = q
        mujoco.mj_kinematics(scene_model, sdata)
        hit = _deepest(scene_model, sdata, probe_geoms, hulls, 0.0, fromto)
        return float(-hit[0]) if hit is not None else 0.0

    def solve_frame(arm_prev, cand, t):
        R_t = frame_at(cand, t)
        ref = qpos_robot[t, cols]
        best = None
        for seed in (arm_prev, ref):
            arm, pe, oe, _R = arm_ik_frame(
                model, data, setup, bases[t].copy(), seed.copy(), ref,
                pocket, cand["anchor_t"][t], R_t)
            if best is None or (pe + ARM_ROT_WEIGHT * oe) < best[3]:
                best = (arm, pe, oe, pe + ARM_ROT_WEIGHT * oe)
        return best[:3]

    def run_pass(cand, stride: int):
        arm = qpos_robot[t0, cols].copy()
        solved: Dict[int, np.ndarray] = {}
        pes, oes, devs, pens = [], [], [], []
        for t in range(t0, T):
            if w[t] == 0.0 and t > t0:
                continue
            if stride > 1 and (t - t0) % stride and t != s:
                continue
            arm, pe, oe = solve_frame(arm, cand, t)
            solved[t] = arm.copy()
            if w[t] == 1.0:
                pes.append(pe)
                oes.append(oe)
                devs.append(np.linalg.norm(arm - qpos_robot[t, cols]))
                pens.append(penetration(t, arm))
        if not pes:
            return (np.inf, np.inf, np.inf, np.inf), solved
        return ((float(np.mean(pes)), float(np.mean(oes)),
                 float(np.mean(devs)), float(np.median(pens))), solved)

    def key_of(pe, oe, dev, pen):
        # a pose that buries the hand in the chair is no grasp at all:
        # the member inside the palm box cages the chair, and the solve
        # then lifts by penetration. Such candidates never win.
        feasible = (pe <= ARM_POS_TOL and oe <= IK_ORI_FLAG
                    and dev <= ARM_DEPARTURE_MAX and pen <= ARM_PEN_TOL)
        return (not feasible,
                dev if feasible else pen + pe + ARM_ROT_WEIGHT * oe)

    best = None   # (cand, key, metrics)
    for cand in cands:
        m, _ = run_pass(cand, stride=3)
        key = key_of(*m)
        if best is None or key < best[1]:
            best = (cand, key, m)
    cand = best[0]
    (pe, oe, dev, pen), solved = run_pass(cand, stride=1)
    if (pe > ARM_POS_TOL or oe > IK_ORI_FLAG or dev > ARM_DEPARTURE_MAX
            or pen > ARM_PEN_TOL):
        return qpos_robot, {"hand": hand, "mode": "reference",
                            "fallback": ("no_fitting_reach"
                                         if pen > ARM_PEN_TOL
                                         else "ik_infeasible"),
                            "pocket_anchor_err": round(pe, 3),
                            "ik_ori_err": round(oe, 3),
                            "hand_penetration_mm": round(pen * 1000, 1),
                            "pocket_anchor_ref": round(ref_dist, 3)}, anchor
    out = qpos_robot.copy()
    for t, arm in solved.items():
        out[t, cols] = (1 - w[t]) * qpos_robot[t, cols] + w[t] * arm
    return out, {"hand": hand, "mode": "ik_retargeted",
                 "fit": cand["name"],
                 "anchor_slide_mm": round(float(np.linalg.norm(
                     cand["anchor"] - anchor)) * 1000, 1),
                 "pocket_anchor_err": round(pe, 3),
                 "ik_ori_err": round(oe, 3),
                 "arm_departure_rad": round(dev, 3),
                 "hand_penetration_mm": round(pen * 1000, 1),
                 "pocket_anchor_ref": round(ref_dist, 3)}, cand["anchor"]


# -------------------------------------------------------------- scene --

def _hull_gids(model) -> List[int]:
    return [g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                ).startswith("largebox_c")]


def _is_hand_geom(name: str) -> bool:
    return bool(re.fullmatch(r"[lr]h(_\w+)?", name))


def _colliding_geoms(model) -> Dict[int, str]:
    """{geom id: name} of every robot geom the hulls can meet."""
    hulls = _hull_gids(model)
    if not hulls:
        return {}
    h = hulls[0]
    out = {}
    for g in range(model.ngeom):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if n.startswith("largebox") or n == "floor":
            continue
        if ((model.geom_contype[g] & model.geom_conaffinity[h])
                or (model.geom_contype[h] & model.geom_conaffinity[g])):
            out[g] = n
    return out


def generate_chair_scene_xml(spec: ChairSpec, mass: float,
                             rest_pos: np.ndarray,
                             rest_quat: np.ndarray) -> str:
    """Template surgery: the largebox becomes the chair — the visual
    mesh plus its convex hulls as collision geoms on one free body.

    Thirty-odd hulls against the robot are too many for explicit
    pairs, so the chair collides by contact bitmask: hulls contype 3 /
    conaffinity 4 meet every robot collision geom (contype 4 /
    conaffinity 2, which still never meet each other or the floor) and
    the floor (default 1/1). A carried chair rests against the body,
    so the template's commented-out body collision geoms (torso,
    pelvis, hips, shins, head, arms) are restored here — the box never
    needed them. The hulls carry the hardened hand-object contact
    parameters (see scene.harden_hand_object_pairs) at priority 1, so
    they win the mix against every robot geom; the template's box
    pairs are dropped."""
    xml = assets.robot_xml()
    sc = spec.mesh_scale
    meshes = [f'<mesh name="chair_visual" file="{spec.obj_file}" '
              f'scale="{sc:.6f} {sc:.6f} {sc:.6f}" />']
    meshes += [f'<mesh name="largebox_hull{i:02d}" file="{h}" '
               f'scale="{sc:.6f} {sc:.6f} {sc:.6f}" />'
               for i, h in enumerate(spec.hulls)]
    xml = mjcf.add_assets(xml, meshes)

    old = '<geom group="3" rgba="0 1 0 1" type="sphere" />'
    assert xml.count(old) == 1, "collision default not found in template"
    xml = xml.replace(old, '<geom group="3" rgba="0 1 0 1" type="sphere" '
                           'contype="4" conaffinity="2" />')
    xml = mjcf.restore_body_collision(xml, BODY_COLLISION_GEOMS)

    mp, mq = spec.mesh_pos, spec.mesh_quat
    place = (f'pos="{mp[0]:.4f} {mp[1]:.4f} {mp[2]:.4f}" '
             f'quat="{mq[0]:.6f} {mq[1]:.6f} {mq[2]:.6f} {mq[3]:.6f}"')
    hull_geoms = "\n      ".join(
        f'<geom name="largebox_c{i:02d}" type="mesh" '
        f'mesh="largebox_hull{i:02d}" {place} group="3" '
        'rgba="0.8 0.6 0.3 0.25" contype="3" conaffinity="4" priority="1" '
        'condim="4" friction="2 0.008 0.0001" solref="0.04 1" '
        'solimp="0.95 0.99 0.001" />'
        for i in range(len(spec.hulls)))
    inertia = mass * spec.inertia_per_mass
    p, q = rest_pos, rest_quat
    body = f'''<body name="largebox" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"
      quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}">
      <freejoint name="largebox_root" />
      <inertial mass="{mass:.3f}"
        pos="{spec.com[0]:.4f} {spec.com[1]:.4f} {spec.com[2]:.4f}"
        diaginertia="{inertia[0]:.3e} {inertia[1]:.3e} {inertia[2]:.3e}" />
      <geom name="largebox_visual" type="mesh" mesh="chair_visual"
        {place} class="visual" material="black" rgba="0.55 0.35 0.15 1" />
      {hull_geoms}
    </body>'''
    xml = mjcf.replace_object_body(xml, body)
    xml = mjcf.drop_hand_object_pairs(xml)
    xml, n = re.subn(r'\s*<pair name="largebox_floor"[^>]*?/>', "", xml)
    assert n == 1, "largebox_floor pair not found in template"
    # bitmask contacts are invisible to the pair count
    return mjcf.set_contact_buffers(xml, geom_pairs=400, contact_points=200)


# --------------------------------------------------------------- trial --

def _hull_gap_fn(hulls: List[int]) -> Callable:
    """pole._conform_fingers gap function against the chair hulls."""
    fromto = np.zeros(6)

    def gap(model, data, gids) -> float:
        best = 1.0
        for gid in gids:
            for g in hulls:
                d = mujoco.mj_geomDistance(model, data, gid, g, best, fromto)
                if d < best:
                    best = d
        return float(best)
    return gap


def _deepest(model, data, geoms, hulls, margin, fromto):
    """(distance, fromto) of the deepest geom-vs-hull overlap under
    `margin`, or None."""
    deepest = None
    for gid in geoms:
        for g in hulls:
            d = mujoco.mj_geomDistance(model, data, gid, g, margin, fromto)
            if d < margin and (deepest is None or d < deepest[0]):
                deepest = (d, fromto.copy())
    return deepest


def place_rest(model, qpos: np.ndarray, s: int) -> Tuple[np.ndarray, Dict]:
    """(dxy, stats): nudge of the resting chair (frames before the
    first grip) that keeps the robot's approach from sweeping through
    it — scene.optimize_box_placement's job, done with the real hulls.

    Every geom the chair can collide with is checked over the frames
    before the first CONTACT (hands and wrists exempt in the last
    HAND_ENGAGE_FRAMES, when they legitimately close on it); the
    smallest xy shift that clears the sweep wins, ties to the smaller
    shift."""
    stats = {"max_pen_before": 0.0, "max_pen_after": 0.0}
    if s <= 0:
        return np.zeros(2), stats
    hulls = _hull_gids(model)
    geoms = _colliding_geoms(model)
    data = mujoco.MjData(model)
    fromto = np.zeros(6)
    frames = list(range(0, s, 3))
    if frames[-1] != s - 1:
        frames.append(s - 1)

    def worst(d: np.ndarray, stop: float = np.inf) -> float:
        pen = 0.0
        for t in frames:
            active = [g for g, n in geoms.items()
                      if not (t >= s - HAND_ENGAGE_FRAMES
                              and (_is_hand_geom(n) or "wrist" in n))]
            data.qpos[:] = qpos[t]
            data.qpos[-7:-5] += d
            mujoco.mj_kinematics(model, data)
            hit = _deepest(model, data, active, hulls, 0.0, fromto)
            if hit is not None:
                pen = max(pen, -hit[0])
                if pen >= stop:
                    return pen
        return pen

    nominal = worst(np.zeros(2))
    stats["max_pen_before"] = round(float(nominal), 4)
    if nominal <= REST_TOL:
        stats["max_pen_after"] = stats["max_pen_before"]
        return np.zeros(2), stats
    n = int(round(2 * REST_SEARCH / REST_STEP)) + 1
    grid = np.linspace(-REST_SEARCH, REST_SEARCH, n)
    cands = sorted(((dx, dy) for dx in grid for dy in grid),
                   key=lambda c: np.hypot(*c))
    best, best_pen = np.zeros(2), nominal
    for c in cands[1:]:
        pen = worst(np.array(c), stop=best_pen)
        if pen < best_pen - 1e-6:
            best, best_pen = np.array(c), pen
            if pen <= REST_TOL:
                break
    stats["max_pen_after"] = round(float(best_pen), 4)
    stats["shift"] = [round(float(v), 4) for v in best]
    return best, stats


def _clear_body(model, qpos: np.ndarray, s: int, e: int, hands, windows,
                margin: float = BODY_CLEAR_MARGIN, floor_z: float = 0.0):
    """Slide the touched or held chair out of any palm or body geom it
    passes through over [s, e]. Returns the per-frame offset (T, 3) and
    the deepest overlap it had to clear.

    The retarget places palms on the human's contact patches, not on
    the G1's palm box, and a chair carried against a human hip sits
    inside the G1's torso capsule; the physics would shove it out at
    once. Per held frame the deepest overlap of a holding palm or any
    body geom with the hulls is found with mj_geomDistance and the
    chair pushed out along the contact direction — pole._clear_palms
    with the hull's closest-point pair standing in for the capsule
    geometry, over the whole body. Fingers are left to the conform
    pass: they are meant to touch. Smoothed for continuity, capped at
    BODY_SHIFT_MAX, and never below the floor."""
    hulls = _hull_gids(model)
    geoms = _colliding_geoms(model)
    palms = {g: windows[h] for g, n in geoms.items()
             for h in hands if n == f"{h[0]}h"}
    body = [g for g, n in geoms.items() if not _is_hand_geom(n)]
    if not hulls or not (palms or body):
        return None, 0.0
    data = mujoco.MjData(model)
    fromto = np.zeros(6)
    T = len(qpos)
    delta = np.zeros((T, 3))
    worst = 0.0
    for t in range(s, e + 1):
        active = body + [g for g, (a, b) in palms.items() if a <= t <= b]
        push = np.zeros(3)
        for _ in range(3):              # several geoms can ask for
            data.qpos[:] = qpos[t]      # different pushes
            data.qpos[-7:-4] += push
            mujoco.mj_kinematics(model, data)
            hit = _deepest(model, data, active, hulls, margin, fromto)
            if hit is None:
                break
            d, ft = hit
            n = ft[:3] - ft[3:]         # robot point - hull point
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                break
            worst = max(worst, -d)
            push = push + (margin - d) * n / nn
        delta[t] = push
    if not np.any(delta):
        return None, 0.0
    delta = smooth3(delta, BODY_CLEAR_SMOOTH)
    mag = np.linalg.norm(delta, axis=1)
    over = mag > BODY_SHIFT_MAX
    delta[over] *= (BODY_SHIFT_MAX / mag[over])[:, None]
    low = qpos[:, -5] + delta[:, 2] - floor_z
    delta[:, 2] -= np.minimum(low, 0.0)
    return delta, worst


def _hold_penetration(model, qpos: np.ndarray, s: int, e: int,
                      stride: int = 3) -> Dict[str, float]:
    """Deepest robot-geom vs chair overlaps over the hold (mm, per
    geom, beyond the spawn tolerance): the retarget's residual the
    physics has to absorb."""
    hulls = _hull_gids(model)
    geoms = _colliding_geoms(model)
    data = mujoco.MjData(model)
    fromto = np.zeros(6)
    worst: Dict[str, float] = {}
    for t in range(s, e + 1, stride):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        for gr, name in geoms.items():
            d = min(mujoco.mj_geomDistance(model, data, gr, g, 0.0, fromto)
                    for g in hulls)
            if d < -SPAWN_SKIP_PEN / 3:
                worst[name] = max(worst.get(name, 0.0), -d)
    return {k: round(float(v) * 1000, 1) for k, v in worst.items()}


def _grasp_quality(model, qpos: np.ndarray, hand: str, s: int, e: int,
                   stride: int = 3) -> Dict[str, float]:
    """How the reference hand holds the chair over its grip: median
    number of finger capsules touching the wood (within 5 mm) and the
    palm box's median surface gap (mm)."""
    hulls = _hull_gids(model)
    names = _colliding_geoms(model)
    fingers = [g for g, n in names.items()
               if re.fullmatch(rf"{hand[0]}h_\w+", n)]
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{hand[0]}h")
    if palm < 0 or not hulls:
        return {}
    data = mujoco.MjData(model)
    fromto = np.zeros(6)
    touching, gaps = [], []
    for t in range(s, e + 1, stride):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        touching.append(sum(
            1 for f in fingers
            if min(mujoco.mj_geomDistance(model, data, f, g, 0.01, fromto)
                   for g in hulls) < 0.005))
        gaps.append(min(mujoco.mj_geomDistance(model, data, palm, g, 0.3,
                                               fromto) for g in hulls))
    return {"fingers_touching": float(np.median(touching)),
            "fingers_total": len(fingers),
            "palm_gap_mm": round(float(np.median(gaps)) * 1000, 1)}


class ChairTask:
    """chair as a ReconTask (recon.run)."""

    name = "chair"

    def detect(self, qpos, meta, params, options) -> Interaction:
        if meta.get("object_pose") is None:
            return Interaction(False, "no object trajectory in the clip",
                               "the chair task needs a raw_motion/behave "
                               "export")
        obj = infer_object(Path(meta["file_path"]).stem,
                           str(params["object"]), meta.get("object_name"))
        spec = load_spec(obj)
        info = detect_chair_contact(meta, qpos, spec)
        info = apply_hand_windows(info, meta, qpos,
                                  left=str(params["left"]),
                                  right=str(params["right"]))
        motion = (f"f{info.motion[0]}-f{info.motion[1]}" if info.motion
                  else "static")
        detail = (f"{info.kind} {_windows(info)} {obj} "
                  f"chair-moves {motion} lift {info.lift:.2f} "
                  f"transport {info.transport:.2f} "
                  f"flags [{','.join(info.quality_flags) or '-'}]")
        return Interaction(info.ok, detail, "no chair interaction",
                           info=info, object=obj)

    def build(self, qpos_robot, meta, inter: Interaction, params) -> Built:
        """The scene and reference of one chair trial. mass=0 uses the
        spec's default; grasp_close overrides the calibrated touch
        closure; grasp is the wrist treatment ("auto" = ik_retargeted
        where the hand model exists, falling back per hand; "reference"
        keeps the clip's own wrist)."""
        info: ChairInfo = inter.info
        mass = float(params["mass"])
        grasp_close = float(params["grasp_close"])
        grasp = str(params["grasp"])
        if grasp not in GRASPS:
            raise SystemExit(f"unknown grasp {grasp!r} "
                             f"(use {', '.join(GRASPS)})")
        spec = load_spec(inter.object)
        mass = mass if mass > 0 else spec.mass
        fps = meta["fps"]
        T = len(qpos_robot)
        obj_ref = clip_object_pose(meta)
        assert len(obj_ref) == T, (len(obj_ref), T)
        s, e = info.grab_frame, info.end_frame
        carry = info.kind == "carry"
        released = carry and "released_early" in info.quality_flags
        windows = {h: (g, e2) for h, g, e2 in zip(info.hands, info.hand_grabs,
                                                  info.hand_ends)}

        # 1. object state from the ORIGINAL motion, made contact-consistent
        obj_qpos, rest_pos, obj_flags = build_chair_trajectory(
            obj_ref, info, fps, floor_z=spec.floor_z)

        calib = {h: calibrate_grasp(h, BAR_DIAMETER, POWER_CLOSE_FRACTION,
                                    wrap_radius=BAR_DIAMETER / 2)
                 for h in info.hands}
        grasp_mode = grasp if grasp != "auto" else "ik_retargeted"
        if any(c is None for c in calib.values()):
            grasp_mode = "reference"    # handless template: no wrist to re-aim

        def assemble(model):
            return assemble_qpos(model, qpos_robot, obj_qpos)

        # the chair is in play from the first touch: nudged out of the
        # robot's approach before it, cleared of the body after
        contacts = {h: info.contacts.get(h, w) for h, w in windows.items()}
        c0 = min([c[0] for c in contacts.values()] or [s])

        # 2. the resting chair out of the approach sweep (body geoms are
        # live in this scene, so a shin through a leg would kick it)
        scene_xml = generate_chair_scene_xml(spec, mass, obj_qpos[0, :3],
                                             obj_qpos[0, 3:])
        model = mujoco.MjModel.from_xml_string(scene_xml)
        layout.check_scene(model)
        rest_place = {}
        if carry and c0 > 0:
            dxy, rest_place = place_rest(model, assemble(model), c0)
            if np.any(dxy):
                obj_qpos = obj_qpos.copy()
                obj_qpos[:s, :2] += dxy
                for k in range(1, POS_BLEND + 1):     # decay onto the clip
                    if s + k <= e:
                        obj_qpos[s + k, :2] += dxy * (1.0 - k / POS_BLEND)
            if rest_place["max_pen_after"] > REST_TOL:
                obj_flags.append("approach_sweeps_chair")

        # 3. each grip's anchor: the member centre line under the clip's
        # OWN palm pocket (median over the grip), and that member's axis
        probe = ChairProbe(spec)
        pocket_tracks = {h: palm_grip_track(qpos_robot, h, BAR_DIAMETER / 2, g,
                                            calib[h][1] if calib[h] else None)
                         for h, (g, _e2) in windows.items()}
        anchors: Dict[str, np.ndarray] = {}
        axes: Dict[str, np.ndarray] = {}
        members: Dict[str, Dict] = {}
        rm = robot_model()
        rd = mujoco.MjData(rm)
        for hand, (g, e2) in windows.items():
            local = to_object_frame(pocket_tracks[hand][g:e2 + 1],
                                    obj_qpos[g:e2 + 1])
            anchors[hand], members[hand] = surface_anchor(
                probe, np.median(local, axis=0))
            rd.qpos[:] = layout.to_model(rm, qpos_robot[g:g + 1])[0]
            mujoco.mj_kinematics(rm, rd)
            R_site = rd.site_xmat[rm.site(f"{hand}_palm").id].reshape(3, 3)
            sp = 1.0 if (calib[hand] is None or calib[hand][1][1] >= 0) else -1.0
            approach = _quat_mat(obj_qpos[g, 3:]).T @ (sp * R_site[:, 1])
            axes[hand], member = member_axis(probe, anchors[hand], approach)
            members[hand].update(member)

        # 4. the chair off the body: body geoms are live in this scene, so
        # a chair inside the torso would be shoved out at once. The clip's
        # own palms count as body here (they get re-posed next)
        scene_xml = generate_chair_scene_xml(spec, mass, obj_qpos[0, :3],
                                             obj_qpos[0, 3:])
        model = mujoco.MjModel.from_xml_string(scene_xml)
        body_clear, shift, worst_pen = None, np.zeros(3), 0.0
        if carry:
            delta, worst_pen = _clear_body(model, assemble(model), c0, e,
                                           info.hands, contacts,
                                           floor_z=spec.floor_z)
            if delta is not None:
                obj_qpos = obj_qpos.copy()
                obj_qpos[:, :3] += delta
                shift = np.abs(delta).max(axis=0)
        if worst_pen > 0:
            body_clear = {"max_pen_before_mm": round(float(worst_pen) * 1000, 1),
                          "max_shift_mm": round(float(np.linalg.norm(shift))
                                                * 1000, 1)}
        grasp_flags: List[str] = []
        if float(np.linalg.norm(shift)) >= BODY_SHIFT_MAX - 1e-9:
            grasp_flags.append("body_shift_capped")

        # 5. arms re-posed to hold the chair where it now is. Each hand's
        # contact TYPE follows the member under it: a bar the closed hand
        # can enclose is WRAPPED (plan_grasp_fit, pocket on the member
        # centre line); a wider face is PRESSED with the open palm
        # (plan_press_fit, palm face on the surface) — and the press is
        # also where a wrap that fits nowhere, or reaches nowhere, lands
        grasp_infos: List[Dict] = []
        kinds: Dict[str, str] = {}
        pockets: Dict[str, Optional[np.ndarray]] = {}

        def _fit_summary(fits):
            return {"candidates": len(fits),
                    "fitting": int(sum(f["penetration"] <= ARM_PEN_TOL
                                       for f in fits)),
                    "best": ({"name": fits[0]["name"],
                              "penetration_mm": round(fits[0]["penetration"]
                                                      * 1000, 1),
                              "slide_mm": round(fits[0]["slide"] * 1000, 1),
                              "turn_deg": round(np.degrees(fits[0]["turn"]), 1)}
                             if fits else None)}

        for hand, (g, e2) in windows.items():
            member, cross = member_kind(probe, members[hand]["hull"])
            members[hand]["kind"] = member
            members[hand]["cross_cm"] = [round(float(c) * 100, 1) for c in cross]
            sp = 1.0 if (calib[hand] is None or calib[hand][1][1] >= 0) else -1.0
            face_pocket = palm_face_pocket(hand, sp) if calib[hand] else None
            kinds[hand] = "wrap" if member == "bar" else "press"
            pockets[hand] = (calib[hand][1] if calib[hand] and member == "bar"
                             else face_pocket)
            ginfo = None
            if grasp_mode == "ik_retargeted" and member == "bar":
                qg = assemble(model)[g].copy()
                wrap = grasp_close if grasp_close > 0 else calib[hand][0]
                finger_pose = {adr: float(angle[0]) for adr, angle in
                               _closure_profile(model, hand, np.array([wrap]),
                                                POWER_CLOSE_FRACTION, wrap=wrap,
                                                caps=calib[hand][2]).items()}
                for adr, angle in finger_pose.items():
                    qg[adr] = angle
                fits = plan_grasp_fit(model, qg, hand, calib[hand][1],
                                      anchors[hand], axes[hand])
                qpos_try, ginfo, anchor_try = retarget_chair_hand(
                    qpos_robot, hand, calib[hand][1], obj_qpos, anchors[hand],
                    axes[hand], g, e2, fps, scene_model=model, fits=fits,
                    finger_pose=finger_pose)
                ginfo["fit_search"] = _fit_summary(fits)
                if ginfo["mode"] == "ik_retargeted":
                    qpos_robot, anchors[hand] = qpos_try, anchor_try
                    kinds[hand], pockets[hand] = "wrap", calib[hand][1]
                else:
                    ginfo = {"wrap_fallback": ginfo["fallback"],
                             "wrap_fit_search": ginfo["fit_search"]}
            if grasp_mode == "ik_retargeted" and face_pocket is not None and (
                    ginfo is None or "wrap_fallback" in ginfo):
                # the press: palm face onto the surface under it
                qg = assemble(model)[g].copy()
                open_pose = {adr: 0.0 for adr in _closure_profile(
                    model, hand, np.zeros(1), POWER_CLOSE_FRACTION)}
                for adr in open_pose:
                    qg[adr] = 0.0
                local = to_object_frame(
                    palm_grip_track(qpos_robot, hand, 0.0, g, face_pocket)[g:e2 + 1],
                    obj_qpos[g:e2 + 1])
                pmed = np.median(local, axis=0)
                d0, _h, surf = probe.closest(pmed)
                n = pmed - surf
                n = (np.sign(d0) * n / np.linalg.norm(n)
                     if np.linalg.norm(n) > 1e-6 else np.array([0.0, 0.0, 1.0]))
                fits = plan_press_fit(model, qg, hand, face_pocket, probe,
                                      surf, n)
                qpos_try, pinfo, anchor_try = retarget_chair_hand(
                    qpos_robot, hand, face_pocket, obj_qpos, surf, n, g, e2,
                    fps, scene_model=model, fits=fits, finger_pose=open_pose)
                pinfo["press_fit_search"] = _fit_summary(fits)
                if ginfo:
                    pinfo.update(ginfo)
                ginfo = pinfo
                if pinfo["mode"] == "ik_retargeted":
                    qpos_robot, anchors[hand] = qpos_try, anchor_try
                    kinds[hand], pockets[hand] = "press", face_pocket
                else:
                    grasp_flags.append(f"ik_fallback_{hand}")
                    kinds[hand] = "reference"
                    if member == "face":
                        anchors[hand] = surf
            if ginfo is None:
                ginfo = {"hand": hand, "mode": "reference"}
                kinds[hand] = "reference"
                if calib[hand] is None:
                    ginfo["fallback"] = "no_hand_model"
            ginfo["hand"] = hand
            ginfo["type"] = kinds[hand]
            ginfo["member"] = members[hand]
            grasp_infos.append(ginfo)

        scene_xml = generate_chair_scene_xml(spec, mass, obj_qpos[0, :3],
                                             obj_qpos[0, 3:])
        model = mujoco.MjModel.from_xml_string(scene_xml)
        qpos = assemble(model)

        # 6. fingers: the calibrated bar closure, then conformed onto the
        # hulls where each finger actually meets the wood
        closures: Dict[str, Dict] = {}
        pre_gaps: Dict[str, Dict] = {}
        gap_fn = _hull_gap_fn(_hull_gids(model))
        for hand, (hand_grab, hand_end) in windows.items():
            if calib[hand] is None:
                closures[hand] = {"touch": 0.0, "hold": 0.0, "caps": None}
                continue
            wrapping = kinds[hand] == "wrap" or (
                kinds[hand] == "reference" and members[hand]["kind"] == "bar")
            wrap = (grasp_close if grasp_close > 0 else calib[hand][0]) \
                if wrapping else 0.0
            hold_c = wrap + WRAP_SQUEEZE if wrapping else 0.0
            _close_hand(model, qpos, hand, hand_grab, hand_end, wrap, hold_c,
                        calib[hand][2] if wrapping else None, hand_end < T - 6)
            closures[hand] = {"touch": wrap, "hold": hold_c}
            # fingers may open fully and never stay buried: a member inside
            # a finger cages the chair as surely as one inside the palm. A
            # pressing hand starts open and curls only until it meets the
            # face
            pre_gaps[hand] = _conform_fingers(model, qpos, hand, hand_grab,
                                              hand_end, gap_fn=gap_fn, hold=0.0,
                                              full_scan=True)

        # the pockets as finally posed, for the residual the grip reward
        # will have to close
        pocket_tracks = {h: palm_grip_track(qpos_robot, h, BAR_DIAMETER / 2, g,
                                            pockets[h])
                         for h, (g, _e2) in windows.items()}
        # what the grip reward will have to close: the pocket's residual off
        # its anchor on the settled chair, from the clip's own wrists
        residual: Dict[str, float] = {}
        for hand, (g, e2) in windows.items():
            Rs = _quat_mats(obj_qpos[g:e2 + 1, 3:])
            anchor_w = obj_qpos[g:e2 + 1, :3] + np.einsum("tij,j->ti", Rs,
                                                          anchors[hand])
            residual[hand] = round(float(np.median(np.linalg.norm(
                pocket_tracks[hand][g:e2 + 1] - anchor_w, axis=1))) * 1000, 1)

        for hand, (hand_grab, hand_end) in windows.items():
            if calib[hand] is None:
                continue
            caps = calib[hand][2] if kinds[hand] != "press" else None
            closures[hand] = {
                "touch": round(closures[hand]["touch"], 4),
                "hold": round(closures[hand]["hold"], 4),
                "caps": ({k: round(float(v), 3) for k, v in caps.items()}
                         if caps else None),
                "pre_conform_gap_mm": pre_gaps.get(hand) or None}

        # the residual the physics has to absorb, how the reference hand
        # holds the chair, and what the carry rests on besides the hands
        hold_pen = _hold_penetration(model, qpos, s, e) if carry else {}
        quality = {h: _grasp_quality(model, qpos, h, g, e2)
                   for h, (g, e2) in windows.items()}
        for h, q in quality.items():
            if q and q.get("palm_gap_mm", 0.0) < -ARM_PEN_TOL * 1000:
                grasp_flags.append(f"palm_buried_{h}")
        supports = detect_supports(model, qpos, s, e) if carry else []
        if supports:
            grasp_flags.append("body_supported")

        # contact reference: each palm at its own reference FK track
        # (the driver's default) over its contact window (a sit labels
        # the hands' touches instead)
        labelled = dict(info.contacts) if carry else dict(info.touches)
        for hand, (g, e2) in windows.items():
            labelled.setdefault(hand, (g, e2))
        contact = np.zeros((T, 2))
        for hand, (g, e2) in labelled.items():
            contact[g:e2 + 1, 0 if hand == "left" else 1] = 1.0

        # interaction graph: per-hand object edges, plus the pelvis on the
        # seat for a sit
        graspish = None
        if carry:
            graspish = GraspInfo(
                pick_frame=s, release_frame=e, box_width=2 * spec.seat_half[1],
                raw_carry_gap=2 * spec.seat_half[1], lift_height=info.lift,
                quality_flags=list(info.quality_flags),
                starts_held=info.starts_held)
        graph = build_interaction_graph(meta, graspish)
        graph.link_contact[:, :2, 1] = 0.0
        graph.edges = [ed for ed in graph.edges if ed.scene != "object"]
        for hand, (g, e2) in labelled.items():
            li = 0 if hand == "left" else 1
            graph.link_contact[g:e2 + 1, li, 1] = 1.0
            graph.edges.append(ContactEdge(
                KEY_LINKS[li], "object", g, e2,
                graph.link_pos[g:e2 + 1, li].mean(axis=0)))
        if info.sit is not None:
            ss, se = info.sit
            graph.link_contact[ss:se + 1, 4, 1] = 1.0
            graph.edges.append(ContactEdge(
                "pelvis", "object", ss, se,
                graph.link_pos[ss:se + 1, 4].mean(axis=0)))
        for sup in supports:            # a hip carry is a pelvis-object edge
            if sup["geom"] in ("pelvis_collision", "left_hip_collision",
                               "right_hip_collision"):
                graph.link_contact[sup["start"]:sup["end"] + 1, 4, 1] = 1.0
                graph.edges.append(ContactEdge(
                    "pelvis", "object", sup["start"], sup["end"],
                    graph.link_pos[sup["start"]:sup["end"] + 1, 4].mean(axis=0)))

        pad = 0 if (released or not carry) else int(HOLD_PAD_S * fps)

        tspec = TrialSpec(
            task_type="chair_carry" if carry else "chair_sit",
            object=ObjectSpec(
                kind="mesh",
                size=[2 * float(spec.seat_half[0]),
                      2 * float(spec.seat_half[1]), spec.height],
                mass=mass,
                info={"name": spec.name, "category": "chair",
                      "height": spec.height, "seat_z": spec.seat_z,
                      "floor_z": round(spec.floor_z, 4),
                      "hulls": len(spec.hulls)}),
            window=(s, e) if carry else (None, None),
            starts_held=info.starts_held,
            lift_height=info.lift,
            # per-hand grip anchors for the solve: hold the SIMULATED chair
            # so its point `anchor` (chair frame) sits in each palm's pocket
            grips=[Grip(
                hand=h, window=(g, e2), pocket=pockets[h], anchor=anchors[h],
                extra={"type": kinds[h],
                       "contact_window": list(labelled[h]),
                       "member_axis": [round(float(v), 4) for v in axes[h]],
                       "closure_touch": closures[h]["touch"],
                       "closure_hold": closures[h]["hold"],
                       "closure_caps": closures[h]["caps"],
                       "pre_conform_gap_mm":
                           closures[h].get("pre_conform_gap_mm"),
                       "pocket_residual_mm": residual.get(h),
                       "reference_grasp": quality.get(h) or None})
                for h, (g, e2) in windows.items()],
            grasp=grasp_infos,
            # body geoms the chair rests on, with their windows: the
            # solve's support term keeps those contacts
            supports=supports,
            flags=info.quality_flags + obj_flags + grasp_flags,
            hand_noise_scale=HAND_NOISE_SCALE,
            extra={"hands": list(info.hands),
                   "kind": info.kind,
                   "object_source": "clip",
                   "touches": [{"hand": h, "start": g, "end": e2}
                               for h, (g, e2) in info.touches.items()],
                   "sit": list(info.sit) if info.sit is not None else None,
                   "motion": (list(info.motion) if info.motion is not None
                              else None),
                   "rest_placement": rest_place or None,
                   "body_clearance": body_clear,
                   "hold_penetration_mm": hold_pen,
                   "transport": round(info.transport, 3)},
        )
        return Built(scene_xml, model, qpos, tspec, contact=contact,
                     graph=graph, pad=pad)

    def describe(self, task_info: Dict) -> str:
        line = ""
        clear = task_info.get("body_clearance") or {}
        if clear:
            line += (f" body-clear (pen {clear['max_pen_before_mm']:.0f}mm "
                     f"cleared, shift {clear['max_shift_mm']:.0f}mm)")
        place = task_info.get("rest_placement") or {}
        if place.get("shift"):
            line += (f" rest-shift {np.hypot(*place['shift']) * 100:.0f}cm "
                     f"(sweep {place['max_pen_before'] * 1000:.0f}mm to "
                     f"{place['max_pen_after'] * 1000:.0f}mm)")
        pen = task_info.get("hold_penetration_mm") or {}
        if pen:
            line += f" HOLD-PEN {max(pen.values()):.0f}mm ({len(pen)} geoms)"
        for g, gi in zip(task_info.get("grips", []),
                         task_info.get("grasp", [])):
            q = g.get("reference_grasp") or {}
            if q:
                line += (f" {g['hand'][0]}:{g.get('type', '?')}"
                         f" {q['fingers_touching']:.0f}/{q['fingers_total']}f"
                         + ("" if gi.get("mode") == "ik_retargeted"
                            else f"({gi.get('fallback', 'reference')})"))
        sups = task_info.get("supports") or []
        if sups:
            frames: dict = {}
            for s_ in sups:
                frames[s_["geom"]] = (frames.get(s_["geom"], 0)
                                      + s_["end"] - s_["start"] + 1)
            top = sorted(frames, key=frames.get, reverse=True)
            line += " supports " + ",".join(
                f"{g.replace('_collision', '')}({frames[g]}f)" for g in top[:4])
            if len(top) > 4:
                line += f"(+{len(top) - 4})"
        return line


def _windows(info: ChairInfo) -> str:
    """hands with their windows, sit and touches, for the summary line."""
    parts = [f"{h}@f{g}-f{e}"
             + (f"(touch f{c[0]}-f{c[1]})"
                if (c := info.contacts.get(h)) and c != (g, e) else "")
             for h, g, e in zip(info.hands, info.hand_grabs, info.hand_ends)]
    parts += [f"{h}~f{s}-f{e}" for h, (s, e) in info.touches.items()]
    if info.sit is not None:
        parts.append(f"sit@f{info.sit[0]}-f{info.sit[1]}")
    return "+".join(parts) or "-"


TASK = ChairTask()


def emit_chair_trial(
    qpos_robot: np.ndarray,
    meta: Dict,
    info: ChairInfo,
    out_root: Path,
    task: str,
    data_id: int = 0,
    object_name: str = "chairwood",
    mass: float = 0.0,
    grasp_close: float = 0.0,
    grasp: str = "auto",
) -> Optional[Path]:
    """Write a complete chair trial from an already detected contact
    (the recon.run pipeline from `build` on). Returns the trial dir, or
    None when the motion admits no scene."""
    from .run import emit  # late: run imports this module's siblings
    params = {"object": object_name, "mass": mass, "grasp_close": grasp_close,
              "grasp": grasp, "left": "auto", "right": "auto"}
    trial, _info, _why = emit(TASK, qpos_robot, meta,
                              Interaction(True, "", info=info,
                                          object=object_name),
                              out_root, task, params, data_id=data_id)
    return trial
