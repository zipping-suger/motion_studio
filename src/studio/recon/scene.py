"""SceneBot-style hindsight scene reconstruction (arXiv 2606.27581, Alg. 1
Stage 3), generalized for arbitrary motion.

From a robot-only kinematic motion + its scene-interaction graph:
  - a box object (freejoint body) sized from the carry-phase hand gap,
  - terrain plateaus: the support slab under the box plus one per
    elevated foot/pelvis contact, merged when overlapping at similar
    heights and CARVED where the robot's swept lower body would collide
    (Alg. 1 line 27) rather than skipping the clip,
  - the box reference trajectory,
  - per-link contact labels for all key links x {terrain, object},
which `box_carry.BoxCarryTask` composes into a trial through the shared
pipeline (recon.run / recon.spec).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import assets, mjcf
from .grasp import GraspInfo
from .graph import InteractionGraph
from .loader import KIM_LEFT_HAND_TIP, KIM_RIGHT_HAND_TIP
from .signal import smooth

__all__ = ["BoxSpec", "Plateau", "SceneSpec", "build_object_trajectory",
           "generate_scene_xml", "reconstruct_terrain"]

# --- box placement optimization ---------------------------------------------
# The rest pose is read off noisy hands at one frame, so the box and its slab
# can land inside the robot's spawn pose or pre-pick path, which MuJoCo
# resolves violently at settle. A bounded kinematic grid search over
# (dx, dy, dyaw) clears the pre-pick sweep out of the rest column.
PLACE_SEARCH_XY = 0.12    # m: placement search half-range per axis
PLACE_STEP_XY = 0.03
PLACE_SEARCH_YAW = 0.3    # rad
PLACE_STEP_YAW = 0.1
PLACE_MARGIN = 0.02       # m: air gap the chosen placement keeps
PLACE_PEN_W = 1.0e3       # penetration weight vs deviation-from-nominal
PLACE_YAW_W = 0.25        # dyaw^2 weight relative to dxy^2
PLACE_MAX_PEN_OK = 0.005  # m: residual true penetration above this -> flag
HAND_ENGAGE_FRAMES = 15   # hands may close on the box this long before pick
HAND_KPS = (22, 23, 24, 25, 30, 31, 32, 33)  # kimodo wrist/hand keypoints
BLEND_FRAMES = 12         # frames to decay the shift back onto the carry track
# Arm keypoints (j18+) legitimately dwell AT the box faces pre-pick — the
# box is defined by the hands — so they get a small radius plus a deadzone
# and only stabs well inside count. Body keypoints (j0-j17) keep the strict
# radius: they are the actual spawn-collision failure.
ARM_KP_START = 18
ARM_R = 0.04              # m: arm/hand keypoint radius (G1 palm capsule r)
ARM_TOL = 0.01            # m: arms may skim the faces; deeper than this counts

# --- terrain reconstruction (Alg. 1 Stage 3) --------------------------------
# Carving grid + the lower-body keypoints (j0-j17; hands legitimately reach
# over the support slab) whose swept volume notches the terrain.
CELL = 0.06            # m: carve-grid resolution
LINK_R = 0.08          # m: keypoint clearance radius
PLATEAU_HALF = 0.15    # m: initial square plateau per terrain edge
MERGE_GAP = 0.05       # m: plateaus closer than this merge...
MERGE_DZ = 0.04        # m: ...when their tops are within this
SUPPORT_MIN_FRAC = 0.4  # surviving support area below this -> skip clip
TERRAIN_MIN_FRAC = 0.35  # mostly-carved foot/seat plateau -> prune the edge
# keypoints exempt from carving a plateau while their link legitimately
# rests on it (the planted foot hovers exactly at the plateau top)
EXEMPT_KPS = {
    "left_foot": (5, 6, 7),                    # ankle_pitch/roll, toe
    "right_foot": (12, 13, 14),
    "pelvis": (0, 1, 2, 3, 4, 8, 9, 10, 11),   # pelvis + hip/thigh/knee chain
}


@dataclass
class BoxSpec:
    half_w: float   # x of box frame: along the hand-hand axis
    half_d: float   # y: depth
    half_h: float   # z
    mass: float


@dataclass
class Plateau:
    """One rectangular 2.5D terrain element (Alg. 1 Stage 3)."""
    center: np.ndarray            # (3,) geom center
    half: np.ndarray              # (3,)
    yaw: float
    kind: str                     # "support" (under object) | "terrain" | "seat"


@dataclass
class SceneSpec:
    box: BoxSpec
    box_rest_pos: np.ndarray      # (3,)
    box_rest_yaw: float
    table_center: np.ndarray      # (3,) or None if box starts on the floor
    table_half: np.ndarray        # (3,)
    table_yaw: float              # slab is yaw-aligned with the box
    has_table: bool
    plateaus: Optional[List[Plateau]] = None
    flags: List[str] = field(default_factory=list)
    placement: Optional[Dict] = None  # box placement optimization stats


def yaw_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def optimize_box_placement(
    jp: np.ndarray,
    pick: int,
    rest_pos: np.ndarray,
    rest_yaw: float,
    half_w: float,
    half_d: float,
    top_z: float,
    z_lo: float = 0.0,
) -> Tuple[np.ndarray, float, Dict]:
    """Nudge the box rest pose so the robot never sweeps its rest volume.

    Penetration is keypoint-sphere vs the yaw-aligned box footprint over
    the z band [z_lo, top_z] — box plus the slab directly beneath it, not
    a full floor column, since feet under a floating slab are legitimate
    — across all pre-pick frames. Clearing the band fixes spawn-in-
    collision and keeps the protected support cell sweep-free. Arm
    keypoints get the ARM_TOL deadzone and hands are exempt in the last
    HAND_ENGAGE_FRAMES. Returns (dxy, dyaw, stats).
    """
    stats = {"max_pen_before": 0.0, "max_pen_after": 0.0, "checked_pts": 0}
    if pick <= 0:
        return np.zeros(2), 0.0, stats
    n_kp = jp.shape[1]
    pts = jp[:pick].reshape(-1, 3)
    kp = np.tile(np.arange(n_kp), pick)
    fr = np.repeat(np.arange(pick), n_kp)
    exempt = np.isin(kp, HAND_KPS) & (fr >= pick - HAND_ENGAGE_FRAMES)
    in_band = (pts[:, 2] - LINK_R < top_z + PLACE_MARGIN) & (
        pts[:, 2] + LINK_R > z_lo - PLACE_MARGIN
    )
    reach = np.linalg.norm(pts[:, :2] - rest_pos[:2], axis=1) < (
        np.hypot(half_w, half_d) + LINK_R + PLACE_MARGIN + PLACE_SEARCH_XY
    )
    keep = in_band & reach & ~exempt
    pts, kp = pts[keep], kp[keep]
    stats["checked_pts"] = int(len(pts))
    if len(pts) == 0:
        return np.zeros(2), 0.0, stats
    arm = kp >= ARM_KP_START
    r_pt = np.where(arm, ARM_R, LINK_R)          # (P,)
    m_pt = np.where(arm, -ARM_TOL, PLACE_MARGIN)

    nx = int(round(2 * PLACE_SEARCH_XY / PLACE_STEP_XY)) + 1
    ny = int(round(2 * PLACE_SEARCH_YAW / PLACE_STEP_YAW)) + 1
    dx, dy, dw = np.meshgrid(
        np.linspace(-PLACE_SEARCH_XY, PLACE_SEARCH_XY, nx),
        np.linspace(-PLACE_SEARCH_XY, PLACE_SEARCH_XY, nx),
        np.linspace(-PLACE_SEARCH_YAW, PLACE_SEARCH_YAW, ny),
        indexing="ij",
    )
    cand = np.stack([dx.ravel(), dy.ravel(), dw.ravel()], axis=1)  # (M, 3)
    nominal = int(np.argmin(np.abs(cand).sum(axis=1)))

    centers = rest_pos[:2] + cand[:, :2]                    # (M, 2)
    yaws = rest_yaw + cand[:, 2]
    d = pts[None, :, :2] - centers[:, None, :]              # (M, P, 2)
    cy, sy = np.cos(yaws)[:, None], np.sin(yaws)[:, None]
    lx = cy * d[..., 0] + sy * d[..., 1]
    ly = -sy * d[..., 0] + cy * d[..., 1]
    ex, ey = np.abs(lx) - half_w, np.abs(ly) - half_d
    sd_xy = np.hypot(np.maximum(ex, 0), np.maximum(ey, 0)) + np.minimum(
        np.maximum(ex, ey), 0.0
    )                                                        # signed, (M, P)
    pen_xy = r_pt + m_pt - sd_xy
    dz = np.maximum(z_lo - pts[:, 2], pts[:, 2] - top_z)     # signed, (P,)
    pen_z = r_pt + m_pt - dz
    pen = np.clip(np.minimum(pen_xy, pen_z[None, :]), 0.0, None)
    # actionable depth: body sans its buffer, arms already tolerance-reduced
    true_pen = np.clip(pen - np.maximum(m_pt, 0.0), 0.0, None)

    stats["max_pen_before"] = round(float(true_pen[nominal].max()), 4)
    if stats["max_pen_before"] == 0.0:
        return np.zeros(2), 0.0, stats

    cost = (
        PLACE_PEN_W * (pen**2).sum(axis=1)
        + cand[:, 0] ** 2 + cand[:, 1] ** 2
        + PLACE_YAW_W * cand[:, 2] ** 2
    )
    best = int(np.argmin(cost))
    stats["max_pen_after"] = round(float(true_pen[best].max()), 4)
    return cand[best, :2].copy(), float(cand[best, 2]), stats


def build_object_trajectory(
    meta: Dict,
    grasp: GraspInfo,
    box_height: float = 0.20,
    box_depth: float = 0.24,
    box_mass: float = 0.5,
    table_margin: float = 0.01,
    squeeze: float = 0.0,
    floor_snap_below: float = 0.0,
) -> Tuple[np.ndarray, SceneSpec]:
    """Returns obj_qpos (T, 7) [pos, quat wxyz] and the scene geometry spec."""
    jp = meta["joint_positions"]
    lh, rh = jp[:, KIM_LEFT_HAND_TIP], jp[:, KIM_RIGHT_HAND_TIP]
    T = len(jp)
    pick, release = grasp.pick_frame, grasp.release_frame

    mid = 0.5 * (lh + rh)
    mid = np.stack([smooth(mid[:, i], 5) for i in range(3)], axis=1)
    hand_axis = lh - rh
    yaw = np.unwrap(np.arctan2(hand_axis[:, 1], hand_axis[:, 0]))
    yaw = smooth(yaw, 9)   # box x along the hand line: faces meet the hands

    if grasp.starts_held:
        # mid-hold start: no rest phase to infer and nothing pre-pick to
        # clear, so the box just rides the hand midpoint from frame 0.
        # An early release leaves it wherever the hands set it.
        half_w = (grasp.box_width + squeeze) / 2
        obj = np.zeros((T, 7))
        for t in range(T):
            if t <= release:
                obj[t, :3] = mid[t]
                obj[t, 3:] = yaw_quat_wxyz(float(yaw[t]))
            else:
                obj[t] = obj[t - 1]
        scene_flags = ["starts_held"]
        if release < T - 1 and obj[release, 2] - box_height / 2 > 0.03:
            scene_flags.append("release_in_air")
        spec = SceneSpec(
            box=BoxSpec(half_w=half_w, half_d=box_depth / 2,
                        half_h=box_height / 2, mass=box_mass),
            box_rest_pos=obj[0, :3].copy(),
            box_rest_yaw=float(yaw[0]),
            table_center=np.zeros(3),
            table_half=np.zeros(3),
            table_yaw=0.0,
            has_table=False,
            flags=scene_flags,
            placement={"shift": [0.0, 0.0], "dyaw": 0.0,
                       "max_pen_before": 0.0, "max_pen_after": 0.0,
                       "checked_pts": 0},
        )
        return obj, spec

    rest_pos = mid[pick].copy()
    rest_yaw = float(yaw[pick])
    half_h = box_height / 2
    table_top = rest_pos[2] - half_h
    # floor snap: rather than a thin floating slab under a low pick,
    # extend the box to the floor so the palms grip at its centroid
    scene_flags = []
    if table_top < floor_snap_below:
        half_h = float(np.clip(rest_pos[2], 0.08, 0.35))
        rest_pos[2] = max(rest_pos[2], half_h)   # bottom on the floor
        if rest_pos[2] - half_h > 0.01:
            # too high even for the tallest snap box: it will drop at settle
            scene_flags.append("box_floats")
        table_top = 0.0
    has_table = table_top > 0.03
    # squeeze widens the box past the hand gap, so the reference palms
    # press into the faces and rollouts bias toward a firm grip
    half_w = (grasp.box_width + squeeze) / 2
    half_d = box_depth / 2

    dxy, dyaw, place_stats = optimize_box_placement(
        jp, pick, rest_pos, rest_yaw, half_w, half_d,
        top_z=rest_pos[2] + half_h,
        z_lo=max(0.0, table_top - 0.04),  # slab band; box bottom if snapped
    )
    rest_pos[:2] += dxy
    rest_yaw += dyaw
    if place_stats["max_pen_after"] > PLACE_MAX_PEN_OK:
        scene_flags.append("spawn_clearance_unresolved")

    # post-pick the reference rejoins the raw mid-hand track: the shift
    # decays over BLEND_FRAMES so the pick is continuous, not a jump
    offset = rest_pos - mid[pick]
    obj = np.zeros((T, 7))
    for t in range(T):
        if t < pick:
            obj[t, :3] = rest_pos
            obj[t, 3:] = yaw_quat_wxyz(rest_yaw)
        elif t <= release:
            w = max(0.0, 1.0 - (t - pick) / BLEND_FRAMES)
            obj[t, :3] = mid[t] + w * offset
            obj[t, 3:] = yaw_quat_wxyz(float(yaw[t]) + w * dyaw)
        else:
            obj[t] = obj[t - 1]
    # Floating slab, no legs: the reference motions are not scene-aware,
    # so a full-height table would occupy space the legs sweep through
    # (SceneBot carves such regions for the same reason), and it is static
    # anyway. Cut to the box footprint and yaw-aligned, since a larger or
    # axis-aligned slab lets a rotated box overhang and tip.
    slab_half = min(0.02, table_top / 2)
    table_half = np.array(
        [half_w + table_margin, half_d + table_margin, slab_half]
    )
    table_center = np.array([rest_pos[0], rest_pos[1], table_top - slab_half])

    spec = SceneSpec(
        box=BoxSpec(half_w=half_w, half_d=half_d, half_h=half_h, mass=box_mass),
        box_rest_pos=rest_pos,
        box_rest_yaw=rest_yaw,
        table_center=table_center,
        table_half=table_half,
        table_yaw=rest_yaw,
        has_table=has_table,
        flags=scene_flags,
        placement={
            "shift": [round(float(dxy[0]), 4), round(float(dxy[1]), 4)],
            "dyaw": round(float(dyaw), 4),
            **place_stats,
        },
    )
    return obj, spec


def generate_scene_xml(spec: SceneSpec, hand_geom: str = "capsule") -> str:
    """Template surgery on the omomo move_largebox scene: swap the box,
    absolutize meshdir, add the table geom + contact pairs.

    hand_geom: "capsule" (fitted to the rubber-hand mesh, rounded) or
    "mesh" (its convex hull — the OmniRetarget approach)."""
    xml = assets.robot_xml()   # template, meshdir already absolutized

    b = spec.box
    m = b.mass
    ix = m / 3 * (b.half_d**2 + b.half_h**2)
    iy = m / 3 * (b.half_w**2 + b.half_h**2)
    iz = m / 3 * (b.half_w**2 + b.half_d**2)
    p = spec.box_rest_pos
    q = yaw_quat_wxyz(spec.box_rest_yaw)
    box_body = f'''<body name="largebox" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"
      quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}">
      <freejoint name="largebox_root" />
      <inertial mass="{m:.3f}" pos="0 0 0"
        diaginertia="{ix:.5f} {iy:.5f} {iz:.5f}" />
      <geom name="largebox_geom" type="box"
        size="{b.half_w:.4f} {b.half_d:.4f} {b.half_h:.4f}"
        class="visual" material="black" />
    </body>'''
    xml = mjcf.replace_object_body(xml, box_body)

    # the template ships body collision geoms commented out (menagerie-MJX
    # lineage); enable them and pair the ones the box can plausibly hit
    xml = mjcf.restore_body_collision(xml)
    body_geoms = [
        "pelvis_collision",
        "left_hip_collision", "right_hip_collision",
        "left_thigh_collision", "right_thigh_collision",
        "left_shin_collision", "right_shin_collision",
        "torso_collision", "head_collision",
    ]
    # arm segments get grip-level friction + stiffness: hugging is a
    # legitimate carry strategy, not an obstacle
    arm_geoms = [
        "left_shoulder_yaw_collision", "right_shoulder_yaw_collision",
        "left_elbow_yaw_collision", "right_elbow_yaw_collision",
        "left_wrist_collision", "right_wrist_collision",
    ]
    xml = mjcf.insert_pairs(xml, [
        f'<pair name="box_{g}" geom1="largebox_geom" geom2="{g}" '
        f'solref="0.008 1" friction="1 1" condim="3" />'
        for g in body_geoms
    ] + [
        f'<pair name="box_{g}" geom1="largebox_geom" geom2="{g}" '
        f'solref="0.004 1" solimp="0.95 0.99 0.001" friction="2 2" '
        f'condim="3" />'
        for g in arm_geoms
    ])

    # replace the crude r=0.05 hand spheres with geometry fitted to the
    # rubber-hand visual mesh. The BrainCo template ships mesh-fitted palm
    # boxes and finger capsules already, so there is nothing to replace.
    if '<geom name="lh" class="hand_collision" />' in xml:
        for name, side, ysign in (("lh", "left", -1), ("rh", "right", 1)):
            if hand_geom == "mesh":
                # convex hull of the visual mesh; the mount offset matches
                # the visual geom (+y left, -y right)
                hand = (
                    f'<geom name="{name}" class="collision" type="mesh" '
                    f'mesh="{side}_rubber_hand" '
                    f'pos="0.0415 {-ysign * 0.003:.3f} 0" />'
                )
            else:
                # along the palm axis, rounded ends so it cannot dig in.
                # The grasp face sits at |y| ~= 0.048 in the wrist_yaw
                # frame, near the mesh's own 0.045 palm surface.
                y = ysign * 0.008
                hand = (
                    f'<geom name="{name}" class="collision" type="capsule" '
                    f'size="0.04" '
                    f'fromto="0.065 {y:.3f} 0.01 0.155 {y:.3f} 0.01" />'
                )
            xml, n = re.subn(rf'<geom name="{name}" class="hand_collision" />',
                             hand, xml)
            assert n == 1, f"hand geom {name} not found"

    xml = mjcf.harden_hand_object_pairs(xml)

    # --- terrain plateaus (Alg. 1 Stage 3) -----------------------------
    # contact partners by kind (explicit-pair-only scene). The first
    # support rectangle keeps the name "table": the viewers and
    # export_dataset.py regex-parse that geom.
    plateaus = spec.plateaus
    if plateaus is None:
        plateaus = [Plateau(spec.table_center, spec.table_half,
                            spec.table_yaw, "support")] if spec.has_table else []
    PARTNERS = {
        "support": ["largebox_geom", "lh", "rh"],
        "terrain": ["lf0", "lf1", "lf2", "lf3",
                    "rf0", "rf1", "rf2", "rf3", "largebox_geom"],
        "seat": ["pelvis_collision", "left_thigh_collision",
                 "right_thigh_collision", "largebox_geom"],
    }
    counts = {"support": 0, "terrain": 0, "seat": 0}
    geom_lines, pair_lines = [], []
    for p in plateaus:
        i = counts[p.kind]
        counts[p.kind] += 1
        if p.kind == "support":
            name = "table" if i == 0 else f"table_{i}"
        else:
            name = f"{p.kind}_{i}"
        q = yaw_quat_wxyz(p.yaw)
        geom_lines.append(
            f'<geom name="{name}" type="box" '
            f'size="{p.half[0]:.4f} {p.half[1]:.4f} {p.half[2]:.4f}" '
            f'pos="{p.center[0]:.4f} {p.center[1]:.4f} {p.center[2]:.4f}" '
            f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" '
            f'class="visual" material="black" />'
        )
        for g in PARTNERS[p.kind]:
            pair_lines.append(
                f'<pair name="{g}_{name}" geom1="{g}" geom2="{name}" '
                f'solref="0.008 1" friction="1 1" condim="3" />'
            )
    if geom_lines:
        xml = xml.replace(
            '<geom name="floor" size="0 0 0.01" type="plane" material="groundplane" />',
            '<geom name="floor" size="0 0 0.01" type="plane" material="groundplane" />\n    '
            + "\n    ".join(geom_lines),
        )
        xml = mjcf.insert_pairs(xml, pair_lines)
    return mjcf.size_contact_buffers(xml)


def _mask_to_rects(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Greedy maximal-rectangle cover of a boolean cell grid."""
    m = mask.copy()
    nx, ny = m.shape
    rects = []
    while m.any():
        best, best_area = None, 0
        for i0 in range(nx):
            for j0 in range(ny):
                if not m[i0, j0]:
                    continue
                width = ny - j0
                for i1 in range(i0, nx):
                    if not m[i1, j0]:
                        break
                    w = 0
                    while j0 + w < ny and m[i1, j0 + w]:
                        w += 1
                    width = min(width, w)
                    area = (i1 - i0 + 1) * width
                    if area > best_area:
                        best_area = area
                        best = (i0, j0, i1, j0 + width - 1)
        i0, j0, i1, j1 = best
        m[i0 : i1 + 1, j0 : j1 + 1] = False
        rects.append(best)
    return rects


def _carve_plateau(
    plat: Plateau,
    jp_lower: np.ndarray,
    exempt: Optional[np.ndarray] = None,
    protect_xy: Optional[np.ndarray] = None,
    protect_end: int = 0,
) -> Optional[Tuple[List[Plateau], int, float]]:
    """Notch a plateau where the robot sweeps through it (Alg. 1 line 27).

    jp_lower: (T, K, 3) keypoints whose sweep carves; exempt: (T, K) True
    where a keypoint's contact is legitimate (planted foot, seated
    pelvis). protect_xy: world xy that must stay supported until
    protect_end. A sweep through that cell during the rest phase means
    the motion admits no support here, so the clip is skipped (None);
    later sweeps neither skip nor carve it — the cell must persist to
    hold the box at t=0, and explicit contact pairs keep legs from
    colliding with supports anyway.

    Returns (rect plateaus, carved cell count, surviving area fraction).
    """
    cy, sy = np.cos(plat.yaw), np.sin(plat.yaw)
    d = jp_lower[..., :2] - plat.center[:2]
    lx = cy * d[..., 0] + sy * d[..., 1]
    ly = -sy * d[..., 0] + cy * d[..., 1]
    lz = jp_lower[..., 2] - plat.center[2]
    act = np.abs(lz) < plat.half[2] + LINK_R
    if exempt is not None:
        act &= ~exempt
    ts, ks = np.where(act)
    if len(ts) == 0:
        return [plat], 0, 1.0
    hx, hy = lx[ts, ks], ly[ts, ks]

    nx = max(1, int(round(plat.half[0] * 2 / CELL)))
    ny = max(1, int(round(plat.half[1] * 2 / CELL)))
    chx, chy = plat.half[0] / nx, plat.half[1] / ny
    if protect_xy is not None:
        dp = protect_xy - plat.center[:2]
        px = cy * dp[0] + sy * dp[1]
        py = -sy * dp[0] + cy * dp[1]

    keep = np.ones((nx, ny), dtype=bool)
    carved = 0
    for i in range(nx):
        cxc = -plat.half[0] + (2 * i + 1) * chx
        for j in range(ny):
            cyc = -plat.half[1] + (2 * j + 1) * chy
            hit = (np.abs(hx - cxc) < chx + LINK_R) & (
                np.abs(hy - cyc) < chy + LINK_R
            )
            if hit.any():
                if protect_xy is not None and abs(px - cxc) <= chx and \
                        abs(py - cyc) <= chy:
                    if (ts[hit] <= protect_end).any():
                        return None
                    continue  # post-pick sweep: keep the box's support cell
                keep[i, j] = False
                carved += 1
    if carved == 0:
        return [plat], 0, 1.0  # untouched: keep the exact original geometry

    out = []
    for i0, j0, i1, j1 in _mask_to_rects(keep):
        lcx = -plat.half[0] + (i0 + i1 + 1) * chx
        lcy = -plat.half[1] + (j0 + j1 + 1) * chy
        out.append(Plateau(
            center=np.array([
                cy * lcx - sy * lcy + plat.center[0],
                sy * lcx + cy * lcy + plat.center[1],
                plat.center[2],
            ]),
            half=np.array([(i1 - i0 + 1) * chx, (j1 - j0 + 1) * chy,
                           plat.half[2]]),
            yaw=plat.yaw,
            kind=plat.kind,
        ))
    return out, carved, float(keep.mean())


def reconstruct_terrain(
    meta: Dict, graph: InteractionGraph, spec: SceneSpec
) -> Optional[Tuple[List[Plateau], Dict]]:
    """Alg. 1 Stage 3 terrain: a plateau per terrain edge, merged at
    similar heights, carved where the robot collides. The box support
    slab joins as a carvable "support" plateau with its CoM cell
    protected. Returns (plateaus, stats), or None when the robot sweeps
    through the box's own resting spot and the clip must be skipped."""
    jp_lower = meta["joint_positions"][:, :18]
    T = len(jp_lower)
    plateaus: List[Plateau] = []
    stats = {"carved_cells": 0, "support_area_frac": 1.0}

    if spec.has_table:
        # the box occupies its support until the pick, or the whole clip
        obj_edges = [e for e in graph.edges if e.scene == "object"]
        protect_end = min(e.start for e in obj_edges) if obj_edges else T - 1
        support = Plateau(center=spec.table_center.copy(),
                          half=spec.table_half.copy(),
                          yaw=spec.table_yaw, kind="support")
        res = _carve_plateau(support, jp_lower,
                             protect_xy=spec.box_rest_pos[:2],
                             protect_end=protect_end)
        if res is None:
            return None
        rects, carved, frac = res
        if frac < SUPPORT_MIN_FRAC:
            return None
        plateaus += rects
        stats["carved_cells"] += carved
        stats["support_area_frac"] = frac

    # foot/pelvis terrain edges -> square plateaus, then merge
    merged: List[Tuple[Plateau, list]] = []
    for e in graph.terrain_edges:
        half_z = min(0.02, e.height / 2)
        merged.append((Plateau(
            center=np.array([e.pos[0], e.pos[1], e.height - half_z]),
            half=np.array([PLATEAU_HALF, PLATEAU_HALF, half_z]),
            yaw=0.0,
            kind="seat" if e.link == "pelvis" else "terrain",
        ), [e]))
    changed = True
    while changed:
        changed = False
        for a in range(len(merged)):
            for b in range(a + 1, len(merged)):
                pa, ea = merged[a]
                pb, eb = merged[b]
                top_a = pa.center[2] + pa.half[2]
                top_b = pb.center[2] + pb.half[2]
                if abs(top_a - top_b) > MERGE_DZ:
                    continue
                if np.any(np.abs(pa.center[:2] - pb.center[:2])
                          > pa.half[:2] + pb.half[:2] + MERGE_GAP):
                    continue
                lo = np.minimum(pa.center[:2] - pa.half[:2],
                                pb.center[:2] - pb.half[:2])
                hi = np.maximum(pa.center[:2] + pa.half[:2],
                                pb.center[:2] + pb.half[:2])
                top = max(top_a, top_b)
                half_z = min(0.02, top / 2)
                merged[a] = (Plateau(
                    center=np.array([*(0.5 * (lo + hi)), top - half_z]),
                    half=np.array([*(0.5 * (hi - lo)), half_z]),
                    yaw=0.0,
                    kind="seat" if pa.kind == "seat" or pb.kind == "seat"
                    else "terrain",
                ), ea + eb)
                del merged[b]
                changed = True
                break
            if changed:
                break

    stats["pruned_plateaus"] = 0
    for plat, contrib in merged:
        exempt = np.zeros((T, 18), dtype=bool)
        for e in contrib:
            for k in EXEMPT_KPS.get(e.link, ()):
                exempt[max(0, e.start - 2) : e.end + 3, k] = True
        res = _carve_plateau(plat, jp_lower, exempt=exempt)
        if res is None:  # unreachable (no protect), defensive
            continue
        rects, carved, frac = res
        if frac < TERRAIN_MIN_FRAC:
            # Alg. 1 line 15: an edge whose plateau the robot mostly sweeps
            # through outside the interaction interval is infeasible — prune
            # it entirely rather than keep carved fragments
            stats["pruned_plateaus"] += 1
            continue
        plateaus += rects
        stats["carved_cells"] += carved

    return plateaus, stats
