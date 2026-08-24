"""One-hand ground-pick reconstruction: a clip of a person picking a
small object off the floor with one hand -> a physics trial.

grasp.py's bimanual machinery keys on rigid co-movement of the hand
PAIR, which a single-hand pick has none of, so detection here is direct:
the picking hand is the one that dips lowest, the pick is its low-still
moment, and the object — a cube, cylinder or ball small enough for one
BrainCo hand — is placed on the floor under the hand tip at that moment.
From the pick on the object rides the hand tip, and the reference curls
the picking hand's fingers around it, so the emitted ctrl already
encodes the grasp.

Two grasp modes: "reference" keeps the clip's own palm-down wrist;
"ik_retargeted" (the default for cylinders) re-aims the wrist by IK so
the palm meets the object's curved side horizontally with the fingers
wrapping its axis. Either way this is only initialization — the solve's
grip reward holds the SIMULATED object in the palm pocket, so MPPI is
free to adapt.

The trial layout matches scene.emit_trial exactly and the object keeps
the template's largebox names, so the hand-object pairs, the solve and
the evaluator apply verbatim. task_info's ``task_type: ground_pick``
tells the solve loop to skip the bimanual box-face palm reward and use
SPIDER's baked contact reference instead.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from . import assets, layout
from .graph import KEY_LINKS, SCENE_TYPES, build_interaction_graph
from .grasp import GraspInfo, hand_tracks
from .loader import DOF_NAMES, compute_qvel
from .table import robot_model
from .scene import (SPAWN_SKIP_PEN, harden_hand_object_pairs,
                    measure_spawn_penetration, size_contact_buffers,
                    yaw_quat_wxyz)
from .signal import smooth

REACH_MAX = 0.35     # m: the picking hand tip must dip below this
HAND_MARGIN = 0.03   # m: the other hand must stay this much higher
HANDS_COLLIDE_GAP = 0.12  # m: hold-time hand-tip gap below which the
                          # idle hand overlaps the picking one (a
                          # BrainCo hand spans ~0.13 m, tip keypoint
                          # mid-palm). Flag only — the solve tracks that
                          # reference, so its self-collision penalty
                          # then fights the tracking term
LIFT_MIN = 0.15      # m: hand rise after the pick for a credible pick-up
STILL_BAND = 0.03    # m: hand counts as at-the-object within this of its min
CLOSE_RAMP = 6       # frames: finger-closure ramp duration
CLOSE_LEAD = 2       # frames the ramp starts before the pick (pre-shaping)
HOLD_TAIL = 0.7      # s kept after the lift completes (see hold_trimmed)
HAND_NOISE_SCALE = 0.25  # solve-time finger exploration damping (loop.py)
PALM_STANDOFF = 0.02     # m: palm face to the held object's surface
GRIP_SQUEEZE = 0.18      # rad past first touch, held by the servos (the
                         # nominal grasp's two-jaw gripper pinch)
WRAP_SQUEEZE = 0.06      # rad: the ik_retargeted preload. A power wrap
                         # bears on palm + ten finger segments, so much
                         # less overshoot buys the same security
WRAP_RIDE_TOL = 0.005    # m: a wrapping digit grazes early and SLIDES
                         # as it curls, so the cap is the LAST closure
                         # whose overlap stays within this — capping at
                         # first contact froze the thumb open before it
                         # opposed
WRAP_POCKET_FWD = 0.004  # m: wrap pocket shift along the fingers
WRAP_POCKET_OUT = 0.002  # m: extra palm clearance beyond the radius.
                         # Kept SMALL: pushing a fat 6 cm can out far
                         # enough for full thumb opposition turns the
                         # grasp into a fingertip cage. Slimmer objects
                         # get the full wrap naturally.

GRASPS = ("auto", "reference", "ik_retargeted")
SIDE_BLEND_S = 0.4       # s: wrist-orientation blend-in ahead of the pick
HOLD_PAD_S = 1.2         # s of frozen held pose appended to the reference,
                         # so the receding horizon has a full window at
                         # every evaluated frame. Without it the last ~1 s
                         # of a held carry ran on a degenerate window
                         # (carried to 1.04 m, then dropped).
IK_ITERS = 30            # damped-least-squares iterations per frame
IK_DAMPING = 0.05
IK_STEP_MAX = 0.3        # rad per joint per iteration
POCKET_Z_MIN = 0.02      # m: the achieved pocket must clear the floor
IK_ORI_FLAG = 0.35       # rad: orientation error above this is flagged
# side-grasp pitch candidates, most side-like first (0 = palm
# horizontal, pi/2 = palm down)
PITCH_SWEEP = tuple(np.radians((0.0, 15.0, 30.0, 45.0, 60.0, 75.0)))
# yaw offsets around where the clip's palm already faces, nearest first;
# for a vertical cylinder every approach is the same grasp, so yaw goes
# purely on wrist reachability
YAW_SWEEP = tuple(np.radians((0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0)))
# the retarget IK moves ONLY the wrist, so the clip keeps authority over
# the arm — a 7-joint whole-arm solve visibly contorted stooping clips
WRIST_JOINTS = ("wrist_roll", "wrist_pitch", "wrist_yaw")

SHAPES = ("cube", "cylinder", "ball")
# shape="auto" reads the clip name: the prompts name the object and the
# npz itself carries no text
SHAPE_KEYWORDS = (("ball", "ball"), ("sphere", "ball"), ("cube", "cube"),
                  ("box", "cube"), ("bottle", "cylinder"),
                  ("can", "cylinder"), ("cylinder", "cylinder"))

# Closure targets as fractions of the closure command (radians); distal
# joints follow their proximal via the template's equality ratios.
#
# CLOSE_FRACTION is the nominal opposed-thumb gripper pinch. The
# metacarpal fraction is past its range on purpose, so the thumb reaches
# FULL OPPOSITION in the first ~15% of the closure — opposition is a
# pose decision, not part of the squeeze, and a fraction that scaled
# with closure left a diagonal claw whose jaws met at ~110 deg. The rest
# of the closure then flexes thumb and fingers toward each other like
# parallel jaws (pads 158-166 deg apart at touch for 4-6 cm objects).
#
# POWER_CLOSE_FRACTION is the ik_retargeted power wrap: the thumb runs
# AHEAD of the finger curl to wrap a cylinder opposite the fingers.
CLOSE_FRACTION = {"index_proximal": 1.0, "middle_proximal": 1.0,
                  "ring_proximal": 1.0, "pinky_proximal": 1.0,
                  "thumb_proximal": 1.0, "thumb_metacarpal": 10.0}
# same saturating opposition as above (a 2.5 metacarpal fraction left
# the thumb only ~40-60 deg across at the wrap's touch closures, standing
# off beside the cylinder); thumb_proximal 1.3 then drives it AHEAD of
# the finger curl to reach around the far side
POWER_CLOSE_FRACTION = {"index_proximal": 1.0, "middle_proximal": 1.0,
                        "ring_proximal": 1.0, "pinky_proximal": 1.0,
                        "thumb_proximal": 1.3, "thumb_metacarpal": 10.0}


@dataclass
class PickInfo:
    hand: str                   # "left" | "right"
    pick_frame: int
    end_frame: int              # inclusive; T-1 when held to the end
    reach_height: float         # picking hand tip minimum, m
    other_min: float            # the other hand's minimum, m
    lift_height: float          # hand rise from pick to peak, m
    quality_flags: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not {"no_low_reach", "two_hand_reach", "no_lift"} \
            & set(self.quality_flags)


def infer_shape(clip_name: str, shape_param: str) -> str:
    if shape_param != "auto":
        if shape_param not in SHAPES:
            raise SystemExit(f"unknown shape {shape_param!r} "
                             f"(use auto, {', '.join(SHAPES)})")
        return shape_param
    name = clip_name.lower()
    for key, shape in SHAPE_KEYWORDS:
        if key in name:
            return shape
    return "cube"


def detect_pick(meta: Dict) -> PickInfo:
    """The picking hand, its low-still pick moment, and the hold window."""
    lh, rh, hand_gap = hand_tracks(meta)
    fps = meta["fps"]
    T = len(lh)
    lz, rz = smooth(lh[:, 2], 5), smooth(rh[:, 2], 5)
    hand = "left" if lz.min() < rz.min() else "right"
    track, z = (lh, lz) if hand == "left" else (rh, rz)
    reach, other_min = float(z.min()), float((rz if hand == "left" else lz).min())

    flags = []
    if reach > REACH_MAX:
        flags.append("no_low_reach")
    if other_min - reach < HAND_MARGIN:
        flags.append("two_hand_reach")

    # the stillest moment of the low band; ties break toward the lowest
    speed = np.linalg.norm(np.gradient(track, axis=0), axis=1) * fps
    low = np.nonzero(z < reach + STILL_BAND)[0]
    pick = int(low[np.lexsort((z[low], speed[low]))[0]])

    # held to the end unless the hand, HAVING RISEN, returns low again;
    # the rise requirement keeps a slow lift out of a deep squat intact
    risen = np.nonzero(z[pick:] > z[pick] + LIFT_MIN)[0]
    if len(risen) == 0:
        end = T - 1
        flags.append("no_lift")
    else:
        r0 = pick + int(risen[0])
        low_again = np.nonzero(z[r0:] < z[pick] + 0.05)[0]
        end = int(r0 + low_again[0]) if len(low_again) else T - 1
        if end < T - 1:
            flags.append("set_down")

    lift = float(z[pick:end + 1].max() - z[pick])

    # informational: the trial still runs, but the panel shows why the
    # reference hands interleave
    if float(hand_gap[pick:end + 1].min()) < HANDS_COLLIDE_GAP:
        flags.append("hands_collide")

    # trim a hold kept to the raw end: a generated clip's tail is
    # decaying motion whose arm jitter shakes the held object out.
    # Set-downs keep their full length.
    if end == T - 1 and lift >= LIFT_MIN:
        r_hi = pick + int(np.argmax(z[pick:end + 1] >= z[pick] + 0.9 * lift))
        hold_end = min(end, r_hi + int(HOLD_TAIL * fps))
        if hold_end < end:
            flags.append("hold_trimmed")
            end = hold_end
    return PickInfo(hand, pick, end, reach, other_min, lift, flags)


# ------------------------------------------------------------- object --

def object_dims(shape: str, size: float, cyl_height: float):
    """(geom type, size attr, half height, inertia diag) for a solid body."""
    if shape == "cube":
        h = size / 2
        i = size * size / 6
        return "box", f"{h:.4f} {h:.4f} {h:.4f}", h, (i, i, i)
    if shape == "cylinder":
        r, hh = size / 2, cyl_height / 2
        iz = r * r / 2
        ixy = (3 * r * r + cyl_height * cyl_height) / 12
        return "cylinder", f"{r:.4f} {hh:.4f}", hh, (ixy, ixy, iz)
    r = size / 2
    i = 0.4 * r * r
    return "sphere", f"{r:.4f}", r, (i, i, i)


def _closure_profile(model, hand: str, c: np.ndarray,
                     fractions: Dict[str, float] = CLOSE_FRACTION,
                     wrap: Optional[float] = None,
                     caps: Optional[Dict[str, float]] = None,
                     ) -> Dict[int, np.ndarray]:
    """{qpos address: angle(t)} for one hand at closure profile c (T,).

    Proximal joints take their fraction of c; distal joints follow
    through the template's equality ratios. Empty on the handless
    template.

    Past `wrap` the advance is UNSCALED — the squeeze adds the same few
    preload radians everywhere, since scaling it by the power profile's
    thumb fractions would sweep the thumb through the object. `caps`
    bound each joint's wrap-phase angle where ITS finger meets the
    object, so a conforming wrap's short fingers stop early."""
    follow = {}
    for e in range(model.neq):
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT:
            follow[model.eq_obj1id[e]] = (model.eq_obj2id[e],
                                          float(model.eq_data[e, 1]))
    if wrap is not None:
        base, extra = np.minimum(c, wrap), np.maximum(c - wrap, 0.0)
    else:
        base, extra = c, 0.0
    out: Dict[int, np.ndarray] = {}
    for name, frac in fractions.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{hand}_{name}_joint")
        if jid < 0:
            return {}
        lo, hi = model.jnt_range[jid]
        target = frac * base
        if caps is not None and name in caps:
            target = np.minimum(target, caps[name])
        angle = np.clip(target + extra, lo, hi)
        out[int(model.jnt_qposadr[jid])] = angle
        for dj, (pj, ratio) in follow.items():
            if pj == jid:
                dlo, dhi = model.jnt_range[dj]
                out[int(model.jnt_qposadr[dj])] = np.clip(angle * ratio,
                                                          dlo, dhi)
    return out


def _body_id(model, stem: str) -> int:
    """Body id, tolerant of the URDF's _Link (left) / _link (right)."""
    for suffix in ("_Link", "_link", ""):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                stem + suffix)
        if bid >= 0:
            return bid
    return -1


def calibrate_grasp(hand: str, size: float,
                    fractions: Dict[str, float] = CLOSE_FRACTION,
                    wrap_radius: Optional[float] = None):
    """(touch_closure, pocket in palm-site frame) — None on the handless
    template. `fractions` picks the closure family the sweep calibrates.

    An FK sweep of the hand alone. Pinch (wrap_radius None): the
    aperture is the finger-pad-to-thumb-pad distance, and touch is where
    it equals the object size; the pocket is the midpoint of the jaw
    chord, corrected for the unequal pad radii so both jaws reach the
    surface together.

    Wrap (wrap_radius given) also returns per-joint angle caps. The
    pad-to-pad chord is no touch criterion for a wrap — it can equal the
    diameter while finger segments cross the object's volume (an
    aperture-derived reference embedded every finger 16-31 mm). So two
    passes: the POCKET comes from the enclosing pose, then the fingers
    sweep shut onto a cylinder FIXED there (the palm is static under
    finger closure) and each finger caps where its own capsules meet the
    surface. A power wrap conforms — the short pinky stops well before
    the long fingers, which one shared angle would bury."""
    model = robot_model()
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{hand}_palm")
    pad_r = []
    for g in ("index1", "middle1", "ring1", "pinky1", "thumb1"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                f"{hand[0]}h_{g}")
        if gid < 0:
            return None
        pad_r.append(float(model.geom_size[gid, 0]))
    r_f, r_t = float(np.mean(pad_r[:4])), pad_r[4]

    def seg_mid(stem):
        bid = _body_id(model, stem)
        kids = [b for b in range(model.nbody)
                if model.body_parentid[b] == bid]
        tip = data.xpos[kids[0]] if kids else data.xpos[bid]
        return 0.5 * (data.xpos[bid] + tip)

    fingers = ("index", "middle", "ring", "pinky", "thumb")
    finger_gids = {
        f: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                              f"{hand[0]}h_{f}{i}") for i in (0, 1)]
        for f in fingers}

    def wrap_gap(gids, pocket_w, wrap_axis):
        """Smallest surface gap between the capsules and the cylinder
        (pocket_w, wrap_axis, wrap_radius); negative = overlap."""
        gap = np.inf
        for gid in gids:
            half = float(model.geom_size[gid, 1])
            axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
            for s in np.linspace(-half, half, 5):
                p = data.geom_xpos[gid] + s * axis
                v = p - pocket_w
                d_line = np.linalg.norm(v - (v @ wrap_axis) * wrap_axis)
                gap = min(gap, d_line - model.geom_size[gid, 0]
                          - wrap_radius)
        return float(gap)

    def probe(c):
        q = np.zeros(model.nq)
        q[3] = 1.0
        for adr, angle in _closure_profile(model, hand, np.array([c]),
                                           fractions).items():
            q[adr] = angle[0]
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        fmids = [seg_mid(f"{hand}_{f}_distal")
                 for f in ("index", "middle", "ring", "pinky")]
        fmid = np.mean(fmids, axis=0)
        tmid = seg_mid(f"{hand}_thumb_distal")
        aperture = float(np.linalg.norm(fmid - tmid)) - r_f - r_t
        site, R = data.site_xpos[sid], data.site_xmat[sid].reshape(3, 3)
        if wrap_radius is None:
            # the center belongs ON the thumb-to-fingers jaw chord,
            # offset for the unequal pad radii. A 6-point mean (palm
            # face + five pads) sat ~10 mm off it toward the fingers,
            # burying the finger pads while the thumb barely touched.
            u = (fmid - tmid) / np.linalg.norm(fmid - tmid)
            center = 0.5 * (fmid + tmid) + 0.5 * (r_t - r_f) * u
            return aperture, R.T @ (center - site)
        inward = fmid - site
        face = site + PALM_STANDOFF * np.sign(inward @ R[:, 1]) * R[:, 1]
        pocket = R.T @ (np.mean([face, *fmids, tmid], axis=0) - site)
        return aperture, pocket

    if not _closure_profile(model, hand, np.zeros(1), fractions):
        return None
    cs = np.arange(0.1, 1.31, 0.025)
    apertures = np.array([probe(c)[0] for c in cs])
    # the aperture shrinks as the hand closes and matches the object
    # here: the pinch's touch, the wrap's enclosing pose
    below = np.nonzero(apertures <= size)[0]
    touch = float(cs[below[0] - 1] if len(below) and below[0] > 0
                  else cs[-1] if not len(below) else cs[0])
    _, pocket = probe(touch)
    if wrap_radius is None:
        return touch, pocket, None

    # fingers close onto the cylinder FIXED at the pocket (only finger
    # joints move, so pocket and axis are constants of the sweep). Each
    # finger caps where its DISTAL pad meets the surface: a snug wrap's
    # proximal segments rest on the object almost immediately, and
    # capping there would leave the hand open. Their overlap is the
    # preload the free-body physics turns into grip force.
    probe(touch)                      # data <- the enclosing pose
    R = data.site_xmat[sid].reshape(3, 3).copy()
    site = data.site_xpos[sid].copy()
    # the centroid pocket tucks a large-radius object 24 mm INTO the
    # palm box; put its center one radius plus clearance off the face
    gid_palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                 f"{hand[0]}h")
    pc = R.T @ (data.geom_xpos[gid_palm] - site)
    pR = R.T @ data.geom_xmat[gid_palm].reshape(3, 3)
    ext_y = float(np.abs(pR[1]) @ model.geom_size[gid_palm])
    sp = 1.0 if pocket[1] >= 0 else -1.0
    pocket = pocket.copy()
    pocket[0] += WRAP_POCKET_FWD
    pocket[1] = pc[1] + sp * (ext_y + wrap_radius + 0.002
                              + WRAP_POCKET_OUT)
    pocket_w = site + R @ pocket
    gaps = {f: [] for f in fingers}
    for c in cs:
        probe(c)
        for f in fingers:
            gaps[f].append(wrap_gap(finger_gids[f][1:], pocket_w, R[:, 2]))
    touch_c = {}
    for f in fingers:
        ok = np.nonzero(np.array(gaps[f]) >= -WRAP_RIDE_TOL)[0]
        touch_c[f] = float(cs[ok[-1]]) if len(ok) else float(cs[0])
    caps = {name: frac * touch_c[name.split("_")[0]]
            for name, frac in fractions.items()}
    return max(touch_c.values()), pocket, caps


def palm_grip_track(qpos_robot: np.ndarray, hand: str, half_h: float,
                    pick_frame: int, pocket=None) -> np.ndarray:
    """FK of the held object's center per frame (T, 3): the calibrated
    grasp pocket carried in the palm-site frame, or — handless — one
    palm-standoff plus half_h along the palm normal facing down."""
    model = robot_model()
    data = mujoco.MjData(model)
    q = layout.to_model(model, qpos_robot)
    sid = model.site(f"{hand}_palm").id

    out = np.zeros((len(q), 3))
    axes = np.zeros((len(q), 3, 3))
    for t in range(len(q)):
        data.qpos[:] = q[t]
        mujoco.mj_kinematics(model, data)
        out[t] = data.site_xpos[sid]
        axes[t] = data.site_xmat[sid].reshape(3, 3)
    if pocket is not None:
        out += np.einsum("tij,j->ti", axes, pocket)
    else:
        sign = -1.0 if axes[pick_frame, 2, 1] > 0 else 1.0
        out += sign * (PALM_STANDOFF + half_h) * axes[:, :, 1]
    return np.stack([smooth(out[:, i], 5) for i in range(3)], axis=1)


def _rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector (mujoco quaternion route)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def _rot_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit axis."""
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def yaw_rot(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def side_grasp_frame(pocket: np.ndarray, yaw: float, z_sign: float,
                     pitch: float = 0.0) -> np.ndarray:
    """Palm-site world orientation for a side grasp, derived from the
    OBJECT: palm normal along the approach yaw, knuckle line (site z)
    vertical so the fingers wrap the object's vertical axis. z_sign
    picks which way they point around it. pitch tilts the frame toward
    palm-down (0 = pure side grasp, pi/2 = top grasp) for clips whose
    arm cannot bring a horizontal palm to the object's height."""
    approach = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    sp = 1.0 if pocket[1] >= 0 else -1.0   # palm-normal sign per hand
    y = sp * approach
    z = np.array([0.0, 0.0, z_sign])
    R = np.stack([np.cross(y, z), y, z], axis=1)
    if pitch == 0.0:
        return R
    for s in (1.0, -1.0):   # tilt in whichever sense faces the palm DOWN
        R_t = _rot_about(R[:, 0], s * pitch) @ R
        if sp * R_t[2, 1] < -1e-6:
            return R_t
    return R


def wrap_frame_candidates(shape: str, t_off: np.ndarray) -> tuple:
    """Candidate wrap frames for the IK sweep, as (pitch, yaw_offsets,
    z_signs) groups most-preferred first; the search takes the best
    carry within the FIRST group that yields a feasible frame.

    z_sign flips the hand about the palm normal. A cylinder must keep
    the NATURAL sense — thumb end of the knuckle line up, since sweeping
    both let IK feasibility pick the upside-down wrap on some clips —
    which t_off's knuckle-line component gives per hand. Cubes and balls
    have no axis to respect, so both senses stay in play."""
    if shape == "cylinder" and abs(t_off[2]) > 1e-9:
        z_signs = (1.0,) if t_off[2] > 0 else (-1.0,)
    else:
        z_signs = (1.0, -1.0)
    return tuple((pitch, YAW_SWEEP, z_signs) for pitch in PITCH_SWEEP)


def wrist_setup(model, hand: str):
    """(palm site id, wrist qpos addrs, dof addrs, joint ranges,
    body-contract columns): the indices every wrist-only IK needs."""
    sid = model.site(f"{hand}_palm").id
    jids = [model.joint(f"{hand}_{j}_joint").id for j in WRIST_JOINTS]
    qadr = [int(model.jnt_qposadr[j]) for j in jids]
    vadr = [int(model.jnt_dofadr[j]) for j in jids]
    cols = [7 + DOF_NAMES.index(f"{hand}_{j}_joint") for j in WRIST_JOINTS]
    return sid, qadr, vadr, model.jnt_range[jids], cols


def wrist_ik_frame(model, data, sid, qadr, vadr, jrange, base, wrist, R_t):
    """Orientation-only DLS IK of the 3 wrist joints onto one frame's
    target palm frame. Returns (wrist joints, palm site position, palm
    site rotation, orientation error rad); `data` is left at the
    returned pose. Shared by the pick and pole retargets."""
    for _ in range(IK_ITERS):
        base[qadr] = wrist
        data.qpos[:] = base
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        R_c = data.site_xmat[sid].reshape(3, 3)
        err = _rotvec(R_t @ R_c.T)
        if np.linalg.norm(err) < 1e-3:
            break
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, None, jacr, sid)
        J = jacr[:, vadr]
        JJt = J @ J.T + (IK_DAMPING ** 2) * np.eye(3)
        dq = np.clip(J.T @ np.linalg.solve(JJt, err),
                     -IK_STEP_MAX, IK_STEP_MAX)
        wrist = np.clip(wrist + dq, jrange[:, 0], jrange[:, 1])
    # measure the RETURNED wrist: on iteration exhaustion `data` is one
    # DLS step stale
    base[qadr] = wrist
    data.qpos[:] = base
    mujoco.mj_kinematics(model, data)
    R_c = data.site_xmat[sid].reshape(3, 3)
    return wrist, data.site_xpos[sid].copy(), R_c.copy(), \
        float(np.linalg.norm(_rotvec(R_t @ R_c.T)))


def retarget_side_grasp(qpos_robot: np.ndarray, meta: Dict, pick: PickInfo,
                        obj_height: float, pocket: np.ndarray, shape: str):
    """Re-orient the picking hand for a side grasp: from SIDE_BLEND_S
    before the pick the wrist blends out of the clip's orientation into
    the grasp frame and holds it through the carry; a set-down blends
    back. Per-frame orientation-only DLS IK on the 3 WRIST_JOINTS,
    warm-started, so the clip keeps FULL authority over the arm and the
    object spawns under wherever the pocket lands.

    Feasibility picks the frame among the shape's candidates: the most
    side-like one whose pocket lands on the object's body wins, so a
    stooping clip tilts toward palm-down exactly as far as it must.

    Returns (new qpos_robot, quality flags, grasp info dict)."""
    model = robot_model()
    data = mujoco.MjData(model)
    sid, qadr, vadr, jrange, cols = wrist_setup(model, pick.hand)

    lh, rh, _ = hand_tracks(meta)
    tip = lh if pick.hand == "left" else rh
    pf = pick.pick_frame
    # yaw fallback for a palm pointing straight down
    approach = tip[pf, :2] - meta["joint_positions"][pf, 0, :2]
    yaw = float(np.arctan2(approach[1], approach[0]))

    # the grasp yaw follows the pelvis after the pick: the object's
    # vertical axis is a symmetry axis, so every yaw is the same grasp,
    # and a world-pinned one would demand an impossible arm once the
    # clip stands up and turns
    rq = qpos_robot[:, 3:7]
    psi = np.unwrap(np.arctan2(
        2 * (rq[:, 0] * rq[:, 3] + rq[:, 1] * rq[:, 2]),
        1 - 2 * (rq[:, 2] ** 2 + rq[:, 3] ** 2)))

    # the thumb sits at one END of the knuckle line, so one z_sign puts
    # its opposition plane above the object's rim, wrapping thin air.
    # t_off locates that plane so such frames are rejected outright.
    tjb = model.jnt_bodyid[
        model.joint(f"{pick.hand}_thumb_proximal_joint").id] \
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                             f"{pick.hand}_thumb_proximal_joint") >= 0 else -1
    data.qpos[:] = layout.to_model(model, qpos_robot[pf:pf + 1])[0]
    mujoco.mj_kinematics(model, data)
    R_site = data.site_xmat[sid].reshape(3, 3)
    t_off = (R_site.T @ (data.xpos[tjb] - data.site_xpos[sid])
             if tjb >= 0 else np.zeros(3))

    # the wrist cannot move the hand, so anchor the sweep on where the
    # clip's palm ALREADY faces at the pick; the nearest-first yaw
    # offsets then mean "smallest wrist correction first"
    face = (1.0 if pocket[1] >= 0 else -1.0) * R_site[:, 1]
    if np.linalg.norm(face[:2]) > 1e-6:
        yaw = float(np.arctan2(face[1], face[0]))

    T = len(qpos_robot)
    blend = max(int(SIDE_BLEND_S * meta["fps"]), 1)
    t0 = max(pf - blend, 0)
    w = np.zeros(T)
    w[t0:pf + 1] = np.linspace(0.0, 1.0, pf - t0 + 1)
    w[pf:] = 1.0
    if "set_down" in pick.quality_flags:    # release: hand the arm back
        out_end = min(pick.end_frame + blend, T - 1)
        w[pick.end_frame:out_end + 1] = np.linspace(
            1.0, 0.0, out_end - pick.end_frame + 1)
        w[out_end + 1:] = 0.0

    def ik_frame(base, wrist, R_t):
        """One frame's wrist IK -> (wrist joints, achieved pocket world
        position, orientation error rad)."""
        wrist, site, R_c, err = wrist_ik_frame(
            model, data, sid, qadr, vadr, jrange, base, wrist, R_t)
        return wrist, site + R_c @ pocket, err

    bases = layout.to_model(model, qpos_robot)   # (T, model nq)

    def frame_at(cand, t):
        """The orientation target at frame t: the candidate grasp frame
        yawed along with the pelvis (identity at the pick)."""
        yaw_c, pitch, z_sign = cand
        d = psi[t] - psi[pf]
        return yaw_rot(d) @ side_grasp_frame(pocket, yaw_c, z_sign, pitch)

    # feasibility: walk in from t0 exactly like the final pass (a cold
    # single-frame IK strands in minima the walk avoids) and check the
    # achieved pocket HEIGHT at the pick — the object spawns at whatever
    # xy the pocket reaches, so height inside the object's body is the
    # one hard requirement for the fingers to wrap it
    def solve_frame(wrist_prev, cand, t):
        """One frame's IK, best of two seeds: the chained warm start and
        the clip's own wrist pose. A single chained seed is branch-
        sensitive — one bad minimum poisons the rest of the window."""
        R_t = frame_at(cand, t)
        best = None
        for seed in (wrist_prev, qpos_robot[t, cols]):
            wrist, pocket_w, ori = ik_frame(bases[t].copy(), seed.copy(),
                                            R_t)
            if best is None or ori < best[2]:
                best = (wrist, pocket_w, ori)
        return best

    def walk_to_pick(cand):
        wrist = qpos_robot[t0, cols].copy()
        pocket_w, ori_err = None, np.inf
        for t in range(t0, pf + 1):
            wrist, pocket_w, ori_err = solve_frame(wrist, cand, t)
        return float(pocket_w[2]), ori_err

    def full_pass(cand):
        """The complete retarget for one candidate frame -> (blended
        qpos, pocket height at the pick, mean orientation error over the
        fully-weighted window)."""
        out = qpos_robot.copy()
        wrist = qpos_robot[t0, cols].copy()  # warm start, carried forward
        z_pocket, ori_errs = np.inf, []
        for t in range(t0, T):
            # t0 has w == 0 but must still be SOLVED, or the warm-start
            # chain diverges from walk_to_pick's and lands elsewhere
            if w[t] == 0.0 and t > t0:
                continue
            wrist, pocket_w, ori_err = solve_frame(wrist, cand, t)
            if t == pf:
                z_pocket = float(pocket_w[2])
            if w[t] == 1.0:
                ori_errs.append(ori_err)
            out[t, cols] = (1 - w[t]) * qpos_robot[t, cols] + w[t] * wrist
        return out, z_pocket, float(np.mean(ori_errs))

    # pitch-major search, most side-like first: the cheap walk-in filters
    # frames by the pocket landing in the object's height band, then
    # survivors run the whole window (a frame that reaches the pick can
    # still be uncarryable) and the best carry at the lowest tilt wins
    lowest = np.inf
    chosen = None
    for pitch, yaw_sweep, z_signs in wrap_frame_candidates(shape, t_off):
        survivors = []
        for yaw_off in yaw_sweep:
            for z_sign in z_signs:
                cand = (yaw + yaw_off, pitch, z_sign)
                tz_off = float((frame_at(cand, pf) @ t_off)[2])
                z_pocket, ori_err = walk_to_pick(cand)
                lowest = min(lowest, z_pocket)
                # ACHIEVED height, which the clip dictates: pocket on
                # the object and the thumb plane on its body, not the rim
                if (POCKET_Z_MIN <= z_pocket <= obj_height - 0.005
                        and POCKET_Z_MIN <= z_pocket + tz_off
                        <= obj_height - 0.005
                        and ori_err < IK_ORI_FLAG):
                    survivors.append(cand)
        results = []
        for cand in survivors:
            out, z_pocket, ori_err = full_pass(cand)
            if ori_err < IK_ORI_FLAG:
                results.append((ori_err, cand, z_pocket, out))
        if results:
            chosen = min(results, key=lambda r: r[0])
            break
    if chosen is None:
        # no tilt reaches the height band, or no side frame survives the
        # carry: the clip's own orientation is the only grasp left
        return qpos_robot, ["ik_unreachable"], {
            "mode": "reference", "fallback": "ik_unreachable",
            "best_pocket_z": (round(lowest, 4)
                              if np.isfinite(lowest) else None)}
    ori_err, (yaw_c, pitch, z_sign), z_pocket, out = chosen

    flags = ["grasp_tilted"] if pitch >= np.pi / 4 else []
    info = {"mode": "ik_retargeted", "yaw": round(yaw_c, 3),
            "height": round(z_pocket, 4),
            "pitch_deg": round(np.degrees(pitch), 1), "z_sign": z_sign,
            "ik_ori_err": round(ori_err, 3)}
    return out, flags, info


def build_pick_object_trajectory(
    qpos_robot: np.ndarray, meta: Dict, pick: PickInfo, half_h: float,
    pocket=None,
) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
    """(obj_qpos (T, 7), rest_pos, rest_yaw, flags). The object rests on
    the floor under the palm's grasp pocket until the pick, then rides
    the pocket with the offset captured there, holding after end."""
    jp = meta["joint_positions"]
    grip = palm_grip_track(qpos_robot, pick.hand, half_h, pick.pick_frame,
                           pocket)
    pelvis = jp[:, 0]
    T, pf = len(jp), pick.pick_frame

    rest = np.array([grip[pf, 0], grip[pf, 1], half_h])
    approach = grip[pf, :2] - pelvis[pf, :2]
    rest_yaw = float(np.arctan2(approach[1], approach[0]))
    quat = yaw_quat_wxyz(rest_yaw)

    flags = []
    if grip[pf, 2] > 2 * half_h + 0.05:
        flags.append("shallow_reach")   # cavity never got near the object

    offset = rest - grip[pf]
    obj = np.zeros((T, 7))
    obj[:, 3:] = quat                   # one-hand carry: kept level
    for t in range(T):
        if t < pf:
            obj[t, :3] = rest
        elif t <= pick.end_frame:
            obj[t, :3] = grip[t] + offset
        else:
            obj[t, :3] = obj[t - 1, :3]
    return obj, rest, rest_yaw, flags


def generate_pick_scene_xml(shape: str, size: float, cyl_height: float,
                            mass: float, rest_pos: np.ndarray,
                            rest_yaw: float) -> str:
    """Template surgery: the largebox becomes the small pick object (name
    kept, so every hand-object pair applies), hand-object pairs get grip
    params, contact buffers are sized to the pair count.

    Cylinders also get a co-located grip CAPSULE the 22 hand pairs are
    re-pointed to: mujoco_warp's capsule-cylinder CCD underreports depth
    by exactly the cylinder radius in the pinned version and drops most
    deep contacts, sinking solved fingers ~35 mm into the object. The
    cylinder keeps the floor pair, which is primitive either way."""
    xml = assets.robot_xml()
    gtype, size_attr, _, inertia = object_dims(shape, size, cyl_height)
    q = yaw_quat_wxyz(rest_yaw)
    grip = ""
    if gtype == "cylinder":
        r, hh = size / 2, cyl_height / 2
        grip = (f'\n      <geom name="largebox_grip" type="capsule" '
                f'size="{r:.4f} {max(hh - r, 0.005):.4f}" '
                f'contype="0" conaffinity="0" rgba="0 0 0 0" />')
    body = f'''<body name="largebox" pos="{rest_pos[0]:.4f} {rest_pos[1]:.4f} {rest_pos[2]:.4f}"
      quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}">
      <freejoint name="largebox_root" />
      <inertial mass="{mass:.3f}" pos="0 0 0"
        diaginertia="{mass * inertia[0]:.3e} {mass * inertia[1]:.3e} {mass * inertia[2]:.3e}" />
      <geom name="largebox_geom" type="{gtype}" size="{size_attr}"
        class="visual" material="black" rgba="0.72 0.45 0.20 1" />{grip}
    </body>'''
    xml, n = re.subn(r'<body name="largebox".*?</body>', body, xml, flags=re.S)
    assert n == 1, "largebox body not found in template"
    if gtype == "cylinder":
        xml, n = re.subn(
            r'(<pair name="(?:left|right)_hand\w*_object"[^>]*geom2=")'
            r'largebox_geom(")',
            r'\1largebox_grip\2', xml)
        assert n in (2, 22), f"expected 2 or 22 hand pairs, re-pointed {n}"
    xml = harden_hand_object_pairs(xml)
    return size_contact_buffers(xml)


# --------------------------------------------------------------- trial --

def _close_fingers(model, qpos: np.ndarray, pick: PickInfo,
                   wrap: float, hold: float,
                   fractions: Dict[str, float] = CLOSE_FRACTION,
                   caps: Optional[Dict[str, float]] = None) -> None:
    """Curl the picking hand around the object in the reference (BrainCo
    only). Close to the touch closure `wrap` through the pick, then
    squeeze on to `hold` past first contact, so the position servos hold
    real pinch force rather than a cage the object shakes out of. A
    set-down opens back up after the end."""
    T = len(qpos)
    t = np.arange(T)
    w1 = np.clip((t - (pick.pick_frame - CLOSE_LEAD)) / CLOSE_RAMP, 0.0, 1.0)
    w2 = np.clip((t - (pick.pick_frame + CLOSE_RAMP)) / CLOSE_RAMP, 0.0, 1.0)
    c = wrap * w1 + (hold - wrap) * w2
    if "set_down" in pick.quality_flags:    # open back up after placing
        c = c * np.clip(1 - (t - pick.end_frame) / CLOSE_RAMP, 0.0, 1.0)
    for adr, angle in _closure_profile(model, pick.hand, c, fractions,
                                       wrap=wrap, caps=caps).items():
        qpos[:, adr] = angle


def emit_pick_trial(
    qpos_robot: np.ndarray,
    meta: Dict,
    pick: PickInfo,
    out_root: Path,
    task: str,
    data_id: int = 0,
    shape: str = "cube",
    size: float = 0.06,
    cyl_height: float = 0.14,
    mass: float = 0.10,
    grasp_close: float = 0.0,
    grasp: str = "auto",
) -> Optional[Path]:
    """Write a complete ground-pick trial (same layout and file contract
    as scene.emit_trial). Returns the trial dir, or None when frame 0
    spawns the robot deep inside the scene. grasp_close overrides the
    calibrated touch closure; grasp is the mode (auto = ik_retargeted
    for cylinders, reference otherwise)."""
    if grasp not in GRASPS:
        raise SystemExit(f"unknown grasp {grasp!r} "
                         f"(use {', '.join(GRASPS)})")
    _, _, half_h, _ = object_dims(shape, size, cyl_height)
    grasp_mode = grasp if grasp != "auto" else (
        "ik_retargeted" if shape == "cylinder" else "reference")
    # the closure family follows the grasp, and calibration sweeps the
    # same family: power wrap for ik_retargeted, pinch for reference
    fractions = (POWER_CLOSE_FRACTION if grasp_mode == "ik_retargeted"
                 else CLOSE_FRACTION)
    calib = calibrate_grasp(
        pick.hand, size, fractions,
        wrap_radius=size / 2 if grasp_mode == "ik_retargeted" else None)
    pocket = calib[1] if calib else None

    grasp_flags: List[str] = []
    grasp_info: Dict = {"mode": grasp_mode}
    if grasp_mode == "ik_retargeted" and pocket is None:
        grasp_mode = "reference"    # handless template: no wrist to re-aim
        grasp_info = {"mode": "reference", "fallback": "no_hand_model"}
    if grasp_mode == "ik_retargeted":
        qpos_robot, grasp_flags, grasp_info = retarget_side_grasp(
            qpos_robot, meta, pick, 2 * half_h, pocket, shape)
        grasp_mode = grasp_info["mode"]     # may fall back to reference
    if grasp_mode == "reference" and fractions is POWER_CLOSE_FRACTION:
        fractions = CLOSE_FRACTION          # fell back: pinch closure
        calib = calibrate_grasp(pick.hand, size, fractions)
        pocket = calib[1] if calib else None
    caps = calib[2] if calib else None
    wrap = grasp_close if grasp_close > 0 else (calib[0] if calib else 0.0)
    squeeze = WRAP_SQUEEZE if grasp_mode == "ik_retargeted" else GRIP_SQUEEZE
    hold = wrap + squeeze if wrap > 0 else 0.0

    obj_qpos, rest_pos, rest_yaw, obj_flags = build_pick_object_trajectory(
        qpos_robot, meta, pick, half_h, pocket)

    scene_xml = generate_pick_scene_xml(shape, size, cyl_height, mass,
                                        rest_pos, rest_yaw)
    model = mujoco.MjModel.from_xml_string(scene_xml)
    layout.check_scene(model)

    qpos = np.zeros((len(qpos_robot), model.nq))
    qpos[:, :7] = qpos_robot[:, :7]
    qpos[:, layout.body_addr(model)] = qpos_robot[:, 7:]
    qpos[:, -7:] = obj_qpos
    _close_fingers(model, qpos, pick, wrap, hold, fractions, caps)
    fps = meta["fps"]

    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)
    spawn_pen = measure_spawn_penetration(model, data)
    if spawn_pen and max(spawn_pen.values()) > SPAWN_SKIP_PEN:
        return None

    task_dir = (Path(out_root) / "processed" / "kimodo" / "unitree_g1"
                / "humanoid_object" / task)
    trial_dir = task_dir / str(data_id)
    trial_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "scene.xml").write_text(scene_xml)

    qvel = compute_qvel(model, qpos, 1.0 / fps)
    ctrl = layout.ctrl_reference(model, qpos)

    # contact reference for SPIDER's baked contact reward (the bimanual
    # box-face term stays off here). Reference grasp: the picking palm on
    # the object's top. IK-retargeted: the palm at its own reference FK
    # track, consistent with the object riding the pocket. Columns match
    # site_ids.
    T = len(qpos)
    col = 0 if pick.hand == "left" else 1
    contact = np.zeros((T, 2))
    contact[pick.pick_frame:pick.end_frame + 1, col] = 1.0
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
                for s in ("left_palm", "right_palm")]
    assert -1 not in site_ids, "palm sites missing from scene"
    contact_pos = np.zeros((T, 2, 3))
    if grasp_mode == "ik_retargeted":
        d2 = mujoco.MjData(model)
        target = np.zeros((T, 3))
        for t in range(T):
            d2.qpos[:] = qpos[t]
            mujoco.mj_kinematics(model, d2)
            target[t] = d2.site_xpos[site_ids[col]]
    else:
        target = qpos[:, -7:-4] + np.array([0.0, 0.0, half_h])
    contact_pos[:, 0] = target
    contact_pos[:, 1] = target

    # a GraspInfo view of the pick, for the interaction-graph labels
    graspish = GraspInfo(
        pick_frame=pick.pick_frame, release_frame=pick.end_frame,
        box_width=size, raw_carry_gap=size, lift_height=pick.lift_height,
        quality_flags=list(pick.quality_flags))
    graph = build_interaction_graph(meta, graspish)

    # a trial that ends held is padded with HOLD_PAD_S of its frozen
    # final pose (zero velocity), so the receding horizon has a full
    # reference window at every evaluated frame
    Tt = pick.end_frame + 1 if "hold_trimmed" in pick.quality_flags else T
    pad = (int(HOLD_PAD_S * fps)
           if "set_down" not in pick.quality_flags else 0)

    def cut(a, pad_tail=None):
        out = a[:Tt]
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
        link_contact=cut(graph.link_contact).astype(np.float32),
        link_pos=cut(graph.link_pos).astype(np.float32),
    )
    info = {
        "task_type": "ground_pick",
        "ref_dt": 1.0 / fps,
        "contact_site_ids": site_ids,
        "source_npz": meta["file_path"],
        "pick_frame": pick.pick_frame,
        "release_frame": pick.end_frame,
        "starts_held": False,
        "pick_hand": pick.hand,
        "object": {"shape": shape, "size": size,
                   "cyl_height": cyl_height, "mass": mass},
        "closure_touch": round(wrap, 4),
        "closure_hold": round(hold, 4),
        # per-joint caps of a conforming power wrap; None for a pinch
        "closure_caps": ({k: round(float(v), 3) for k, v in caps.items()}
                         if caps else None),
        "hand_noise_scale": HAND_NOISE_SCALE,
        "grasp": grasp_info,
        # the calibrated pocket (palm-site frame): the solve's grip
        # reward holds the SIMULATED object here
        "grasp_pocket": ([round(float(v), 4) for v in pocket]
                         if pocket is not None else None),
        "reach_height": round(pick.reach_height, 4),
        "lift_height": round(pick.lift_height, 4),
        "quality_flags": pick.quality_flags + obj_flags + grasp_flags,
        "has_table": False,
        "table_top_z": 0.0,
        # bounding dims, so box-oriented viewers can still frame it
        "box_size": [size, size,
                     cyl_height if shape == "cylinder" else size],
        "box_mass": mass,
        "spawn_penetration": spawn_pen,
        "key_links": list(KEY_LINKS),
        "scene_types": list(SCENE_TYPES),
        "graph_flags": graph.flags,
        "terrain_geoms": [],
    }
    (task_dir / "task_info.json").write_text(json.dumps(info, indent=2))
    return trial_dir
