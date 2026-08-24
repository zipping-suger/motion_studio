"""Under-table scene reconstruction: head-driven table placement + MJCF
injection.

The collision-free-augmentation sibling of `scene.py`: instead of the box
a carry implies, infer the TABLE an under-table pick implies — a slab + 4
legs placed over the ducking robot from FK of its head trajectory — and
inject it into the G1 scene as a solid obstacle the MPPI solve must avoid
while tracking the reference. Pick mode only.

numpy + mujoco only (studio's own venv). The solve against the injected
scene lives in `studio.solve.mppi_loop` (solve venv).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np

from . import assets, layout
from .loader import compute_qvel

# ------------------------------------------------- robot collision model --

# Whole-body collision geoms activated on top of the template, which ships
# most of them commented out (thigh/wrist are active there as ghost
# self-pair partners and get OVERRIDDEN below). Torso/leg/head specs are
# the template's own; wrist/hand are enlarged to match the visual mesh
# extent.  name -> (parent body, type, size, placement)
BODY_COLLISION_GEOMS: Dict[str, tuple] = {
    "pelvis_collision": ("pelvis", "sphere", "0.07", {"pos": "0 0 -0.08"}),
    "left_hip_collision": ("left_hip_roll_link", "capsule", "0.06",
                           {"fromto": "0.02 0 0 0.02 0 -0.08"}),
    "left_thigh_collision": ("left_hip_yaw_link", "capsule", "0.055",
                             {"fromto": "-0.0 0 -0.03 -0.06 0 -0.17"}),
    "left_shin_collision": ("left_knee_link", "capsule", "0.045",
                            {"fromto": "0.01 0 0 0.01 0 -0.15"}),
    "left_linkage_brace_collision": ("left_knee_link", "capsule", "0.03",
                                     {"fromto": "0.01 0 -0.2 0.01 0 -0.28"}),
    "right_hip_collision": ("right_hip_roll_link", "capsule", "0.06",
                            {"fromto": "0.02 0 0 0.02 0 -0.08"}),
    "right_thigh_collision": ("right_hip_yaw_link", "capsule", "0.055",
                              {"fromto": "-0.0 0 -0.03 -0.06 0 -0.17"}),
    "right_shin_collision": ("right_knee_link", "capsule", "0.045",
                             {"fromto": "0.01 0 0 0.01 0 -0.15"}),
    "right_linkage_brace_collision": ("right_knee_link", "capsule", "0.03",
                                      {"fromto": "0.01 0 -0.2 0.01 0 -0.28"}),
    "torso_collision": ("torso_link", "capsule", "0.09",
                        {"fromto": "0.01 0 0.08 0.01 0 0.2"}),
    "head_collision": ("torso_link", "sphere", "0.06", {"pos": "0 0 .43"}),
    "left_shoulder_yaw_collision": ("left_shoulder_yaw_link", "capsule",
                                    "0.035", {"fromto": "0 0 -0.08 0 0 0.05"}),
    "left_elbow_yaw_collision": ("left_elbow_link", "capsule", "0.035",
                                 {"fromto": "-0.01 0 -0.01 0.08 0 -0.01"}),
    "left_forearm_collision": ("left_elbow_link", "capsule", "0.05",
                               {"fromto": "-0.01 0 -0.01 0.10 0 -0.01"}),
    "left_wrist_collision": ("left_wrist_pitch_link", "capsule", "0.05",
                             {"fromto": "-0.01 0 0 0.06 0 0"}),
    "left_hand_collision": ("left_wrist_yaw_link", "capsule", "0.06",
                            {"fromto": "0.02 0 0 0.14 -0.02 0"}),
    "right_shoulder_yaw_collision": ("right_shoulder_yaw_link", "capsule",
                                     "0.035", {"fromto": "0 0 -0.08 0 0 0.05"}),
    "right_elbow_yaw_collision": ("right_elbow_link", "capsule", "0.035",
                                  {"fromto": "-0.01 0 -0.01 0.08 0 -0.01"}),
    "right_forearm_collision": ("right_elbow_link", "capsule", "0.05",
                                {"fromto": "-0.01 0 -0.01 0.10 0 -0.01"}),
    "right_wrist_collision": ("right_wrist_pitch_link", "capsule", "0.05",
                              {"fromto": "-0.01 0 0 0.06 0 0"}),
    "right_hand_collision": ("right_wrist_yaw_link", "capsule", "0.06",
                             {"fromto": "0.02 0 0 0.14 0.02 0"}),
}

# BrainCo fingertips reach ~0.22 m past the wrist against the rubber
# hand's ~0.15, so stretch the hand slab proxy to cover the whole hand
BRAINCO_HAND_COLLISION = {
    "left_hand_collision": ("left_wrist_yaw_link", "capsule", "0.06",
                            {"fromto": "0.03 0 0 0.17 0 0"}),
    "right_hand_collision": ("right_wrist_yaw_link", "capsule", "0.06",
                             {"fromto": "0.03 0 0 0.17 0 0"}),
}

# every robot geom paired against the table; the SDF reward's FK sample
# points come from the ``*_collision`` subset only
ROBOT_COLLISION_GEOMS = list(BODY_COLLISION_GEOMS) + [
    "lh", "rh",
    "lf0", "lf1", "lf2", "lf3",
    "rf0", "rf1", "rf2", "rf3",
]


def _find_body(element: ET.Element, body_name: str) -> Optional[ET.Element]:
    if element.tag == "body" and element.get("name") == body_name:
        return element
    for child in element:
        found = _find_body(child, body_name)
        if found is not None:
            return found
    return None


def _find_geom(element: ET.Element, geom_name: str) -> Optional[ET.Element]:
    for geom in element.iter("geom"):
        if geom.get("name") == geom_name:
            return geom
    return None


def robot_scene_tree() -> ET.Element:
    """The template as an ET root, robot-only, full collision set.

    Strips the largebox body and every pair referencing it, then adds
    BODY_COLLISION_GEOMS. The template's contype/conaffinity 0 default
    keeps those inert until a <pair> names them, so this model FKs and
    simulates exactly like the template minus the box.
    """
    root = ET.fromstring(assets.robot_xml())
    worldbody = root.find("worldbody")
    contact = root.find("contact")

    box = next(b for b in worldbody.findall("body")
               if b.get("name") == "largebox")
    worldbody.remove(box)
    for pair in list(contact):
        if "largebox_geom" in (pair.get("geom1"), pair.get("geom2")):
            contact.remove(pair)

    specs = dict(BODY_COLLISION_GEOMS)
    if assets.brainco_available():
        specs.update(BRAINCO_HAND_COLLISION)
    for name, (parent_name, gtype, size, place) in specs.items():
        existing = _find_geom(worldbody, name)
        if existing is not None:
            # thigh/wrist ship ACTIVE (ghost self-pair partners): override
            # with this model's own, deliberately larger, spec
            existing.attrib.pop("pos", None)
            existing.attrib.pop("fromto", None)
            existing.set("type", gtype)
            existing.set("size", size)
            for k, v in place.items():
                existing.set(k, v)
            continue
        parent = _find_body(worldbody, parent_name)
        assert parent is not None, f"template lost body {parent_name}"
        ET.SubElement(parent, "geom", {
            "name": name, "class": "collision", "type": gtype, "size": size,
            **place,
        })
    return root


def robot_model() -> mujoco.MjModel:
    """The robot-only G1 model used for FK and trial emission."""
    model = mujoco.MjModel.from_xml_string(
        ET.tostring(robot_scene_tree(), encoding="unicode"))
    layout.check_robot(model)
    return model


# ------------------------------------------------------------ table spec --

def _rot_xy(arr: np.ndarray, ang: float) -> np.ndarray:
    """Rotate points (..., 3) about the world z-axis by ``ang`` (rad)."""
    c, s = np.cos(ang), np.sin(ang)
    out = np.array(arr, dtype=np.float64, copy=True)
    x, y = arr[..., 0], arr[..., 1]
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out


@dataclass
class TableSpec:
    """A legged table: tabletop slab + 4 square legs.

    Boxes are world-frame; ``yaw`` (rad, about world z) applies to every box
    and the leg centers are already rotated with the slab, so the table is
    rigid. slab_half is in the table frame (x = table normal/depth).
    """

    slab_center: np.ndarray  # (3,)
    slab_half: np.ndarray    # (3,) table-frame half extents
    leg_centers: np.ndarray  # (4, 3)
    leg_half: np.ndarray     # (3,) shared cross-section
    yaw: float = 0.0
    meta: dict = field(default_factory=dict)

    def primitives(self) -> list:
        """[(type, pos[3], [hx, hy, hz, yaw]), ...] — slab first, then legs.
        The format the SDF reward and the clearance checks consume."""
        prims = [("box", self.slab_center.tolist(),
                  [*self.slab_half.tolist(), self.yaw])]
        for c in self.leg_centers:
            prims.append(("box", c.tolist(), [*self.leg_half.tolist(), self.yaw]))
        return prims

    def geom_names(self) -> list:
        return ["table_geom"] + [f"table_leg_{i}_geom"
                                 for i in range(len(self.leg_centers))]

    @property
    def underside_z(self) -> float:
        return float(self.slab_center[2] - self.slab_half[2])

    @property
    def x_near(self) -> float:
        """Near-edge coordinate along the table normal (table frame)."""
        return float(_rot_xy(self.slab_center, -self.yaw)[0] - self.slab_half[0])

    @property
    def x_far(self) -> float:
        """Far-edge coordinate along the table normal (table frame)."""
        return float(_rot_xy(self.slab_center, -self.yaw)[0] + self.slab_half[0])


# ------------------------------------------------------------------- FK --

def _head_trajectory(
    qpos_ref: np.ndarray, mj_model: mujoco.MjModel
) -> Tuple[np.ndarray, float]:
    """World head-center positions (T, 3) + sphere radius via FK of the
    ``head_collision`` geom. Kimodo's 34-joint skeleton has no head joint,
    so this sphere (torso frame, see BODY_COLLISION_GEOMS) is the estimate."""
    gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "head_collision")
    assert gid >= 0, "head_collision missing — build the model via robot_model()"
    radius = float(mj_model.geom_size[gid][0])
    qpos_ref = layout.to_model(mj_model, qpos_ref)
    data = mujoco.MjData(mj_model)
    head = np.zeros((len(qpos_ref), 3))
    for t in range(len(qpos_ref)):
        data.qpos[:] = qpos_ref[t]
        mujoco.mj_fwdPosition(mj_model, data)
        head[t] = data.geom_xpos[gid]
    return head, radius


def _palm_trajectory(qpos_ref: np.ndarray, mj_model: mujoco.MjModel) -> np.ndarray:
    """World palm-site positions (T, 2, 3)."""
    left_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
    right_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    qpos_ref = layout.to_model(mj_model, qpos_ref)
    data = mujoco.MjData(mj_model)
    hands = np.zeros((len(qpos_ref), 2, 3))
    for t in range(len(qpos_ref)):
        data.qpos[:] = qpos_ref[t]
        mujoco.mj_fwdPosition(mj_model, data)
        hands[t, 0] = data.site_xpos[left_sid]
        hands[t, 1] = data.site_xpos[right_sid]
    return hands


# arm-chain geoms: with ``arm_conflict`` on they may conflict with the
# slab, so they must not push the table away from the head at placement
_ARM_KEYWORDS = ("shoulder", "elbow", "forearm", "wrist", "hand")


def _collision_geom_entries(mj_model: mujoco.MjModel,
                            extra_geoms: tuple = ()) -> list:
    """(name, gid, radius, is_capsule, half_length) for all ``*_collision``
    geoms, plus any explicitly named ``extra_geoms`` (e.g. the lf*/rf* foot
    pads, which follow a different naming convention)."""
    entries = []
    for gid in range(mj_model.ngeom):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not name or ("_collision" not in name and name not in extra_geoms):
            continue
        r = float(mj_model.geom_size[gid, 0])
        cap = mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CAPSULE
        half_len = float(mj_model.geom_size[gid, 1]) if cap else 0.0
        entries.append((name, gid, r, cap, half_len))
    return entries


# the tiny contact-pad spheres under each foot — the geoms that matter when
# a FOOT is the colliding body part (they carry no "_collision" suffix)
FOOT_GEOMS = ("lf0", "lf1", "lf2", "lf3", "rf0", "rf1", "rf2", "rf3")


def arm_point_mask(mj_model: mujoco.MjModel) -> np.ndarray:
    """Bool mask over trajectory_collision_points columns: True = the point
    belongs to an arm geom (see _ARM_KEYWORDS)."""
    mask = []
    for name, _, _, cap, _ in _collision_geom_entries(mj_model):
        arm = any(k in name for k in _ARM_KEYWORDS)
        mask += [arm, arm, arm] if cap else [arm]
    return np.asarray(mask)


def trajectory_collision_points(
    qpos_arr: np.ndarray, mj_model: mujoco.MjModel,
    extra_geoms: tuple = (),
) -> Tuple[np.ndarray, np.ndarray]:
    """FK sample points + radii for all ``*_collision`` geoms (plus any
    named ``extra_geoms``) over a trajectory.

    Same sphere approximation as the SDF reward: every geom contributes its
    center, capsules additionally both axis endpoints. Returns (T, P, 3)
    points and (P,) radii.
    """
    qpos_arr = layout.to_model(mj_model, np.atleast_2d(qpos_arr))
    entries = _collision_geom_entries(mj_model, extra_geoms)
    radii = []
    for _, _, r, cap, _ in entries:
        radii += [r, r, r] if cap else [r]

    data = mujoco.MjData(mj_model)
    T = len(qpos_arr)
    points = np.zeros((T, len(radii), 3))
    for t in range(T):
        data.qpos[:] = qpos_arr[t]
        mujoco.mj_fwdPosition(mj_model, data)
        row = []
        for _, gid, _, cap, half_len in entries:
            center = data.geom_xpos[gid].copy()
            row.append(center)
            if cap:
                axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
                row.append(center + half_len * axis)
                row.append(center - half_len * axis)
        points[t] = row
    return points, np.asarray(radii)


# --------------------------------------------------------- analytic SDF --

def _point_box_sdf_np(points, center, half_extents, yaw):
    d = points - center
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.stack(
        [d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c, d[:, 2]],
        axis=-1)
    q = np.abs(local) - half_extents
    outside = np.linalg.norm(np.clip(q, 0.0, None), axis=-1)
    inside = np.clip(q.max(axis=-1), None, 0.0)
    return outside + inside


def clearance_profile_prims(
    points: np.ndarray, radii: np.ndarray, prims: list
) -> np.ndarray:
    """Per-frame worst surface clearance (T,) of trajectory points vs
    primitives (format of TableSpec.primitives). Negative = penetrating."""
    T, P, _ = points.shape
    flat = points.reshape(T * P, 3)
    min_sdf = np.full(T * P, np.inf)
    for _prim_type, pos, params in prims:
        sdf = _point_box_sdf_np(flat, np.asarray(pos),
                                np.asarray(params[:3]), params[3])
        min_sdf = np.minimum(min_sdf, sdf)
    return (min_sdf.reshape(T, P) - radii[None, :]).min(axis=1)


# ------------------------------------------------------------ estimation --

def _make_legs(x_near, x_far, cy, hy, h_under, tcfg):
    off = tcfg.leg_inset + tcfg.leg_cross / 2
    leg_half = np.array([tcfg.leg_cross / 2, tcfg.leg_cross / 2, h_under / 2])
    centers = np.array([
        [x_near + off, cy - (hy - off), h_under / 2],
        [x_near + off, cy + (hy - off), h_under / 2],
        [x_far - off, cy - (hy - off), h_under / 2],
        [x_far - off, cy + (hy - off), h_under / 2],
    ])
    return centers, leg_half


def _shift_height_in_place(spec: TableSpec, dz: float) -> float:
    """Shift the slab vertically by dz (negative = lower); legs re-span
    floor->underside. Returns the new underside height."""
    h = spec.underside_z + dz
    if h <= 0.1:
        raise ValueError(f"table underside {h:.3f}m too low (dz={dz:+.3f})")
    spec.slab_center[2] += dz
    spec.leg_half[2] = h / 2
    spec.leg_centers[:, 2] = h / 2
    return h


def estimate_table(
    qpos_ref: np.ndarray,
    mj_model: mujoco.MjModel,
    tcfg,
    rng: Optional[np.random.RandomState] = None,
    yaw: Optional[float] = None,
    verbose: bool = True,
) -> TableSpec:
    """Estimate a legged table above the ducking robot from the head
    trajectory.

    ``mj_model`` must come from robot_model(); ``tcfg`` is a namespace of
    the table params (tasks.under_table_params.SCENE_DEFAULTS); ``yaw``
    (rad) fixes the yaw, None samples from +/-tcfg.yaw_range.

    Placement: the underside sits just above the highest body point of the
    deep phase but well below standing head height, so the duck is a real
    constraint. The near edge is pushed +x (far edge pinned past the
    deepest hand reach) until reference penetration is within
    ``max_ref_table_pen`` — the entry/exit sweep binds, not the deep
    phase. Legs straddle the reference and the table widens until they
    clear it. Finally the slab drops by ``underside_drop`` so the
    reference itself conflicts and the solve must crouch below it.

    Yaw is handled by rotating the FK trajectories into the table frame,
    estimating axis-aligned there, and rotating the boxes back, which
    makes the near-edge scan corner-aware.

    Raises ValueError when the clip has no duck phase — a verdict, not an
    error.
    """
    if rng is None:
        rng = np.random.RandomState()
    warnings: list = []

    psi = float(yaw) if yaw is not None else (
        float(rng.uniform(-tcfg.yaw_range, tcfg.yaw_range))
        if tcfg.yaw_range else 0.0
    )

    head_w, head_r = _head_trajectory(qpos_ref, mj_model)  # world, for meta
    head = _rot_xy(head_w, -psi)  # table frame from here on
    hands = _rot_xy(_palm_trajectory(qpos_ref, mj_model), -psi)
    points_w, radii = trajectory_collision_points(qpos_ref, mj_model)
    points = _rot_xy(points_w, -psi)
    flat = points.reshape(-1, 3)                    # every body point
    radii_flat = np.tile(radii, len(points))

    # with arm_conflict (default) the arm chain is excluded from the
    # placement scans: re-pathing arms is the solver's work, and a reach at
    # underside height would push the near edge until the table escapes the
    # head. Leg clearance and ref_min_clearance still use every point.
    if getattr(tcfg, "arm_conflict", True):
        keep = ~arm_point_mask(mj_model)
        scan_pts, scan_radii = points[:, keep], radii[keep]
    else:
        scan_pts, scan_radii = points, radii
    scan_flat = scan_pts.reshape(-1, 3)
    scan_radii_flat = np.tile(scan_radii, len(scan_pts))

    head_top = head[:, 2] + head_r
    H_stand = float(np.median(head_top[:10]))
    H_min = float(head_top.min())
    deep = head_top <= H_min + tcfg.deep_band
    duck = head_top < tcfg.duck_frac * H_stand
    if not duck.any():
        raise ValueError(
            f"no duck phase found (head top range {H_min:.2f}-"
            f"{head_top.max():.2f}, standing {H_stand:.2f}) — is this an "
            f"under-table motion?"
        )

    # Underside height: clear the deep-phase body, stay a real duck.
    deep_body_top = float((scan_pts[deep, :, 2] + scan_radii[None, :]).max())
    h_lo = tcfg.underside_min
    h_hi = min(tcfg.underside_max, H_stand - tcfg.min_duck - tcfg.slab_thickness)
    h_base = deep_body_top + tcfg.head_clearance \
        + rng.uniform(-tcfg.h_jitter_down, tcfg.h_jitter_up)
    h_base = float(np.clip(h_base, h_lo, h_hi))

    # Reach & lateral extent.
    pick_reach_x = float(hands[duck, :, 0].max())
    x_far = pick_reach_x + tcfg.far_margin
    cy = float(head[deep, 1].mean()) + rng.uniform(-tcfg.y_jitter, tcfg.y_jitter)
    hy = tcfg.table_width / 2

    x_near_seed = float(head[deep, 0].min()) - tcfg.back_start

    def push_near_edge(h: float, x_near: float, hy: float) -> float:
        """Move the near edge +x until worst reference penetration of the
        slab is within max_ref_table_pen (arm chain excluded)."""
        th = tcfg.slab_thickness / 2
        for _ in range(30):
            center = np.array([(x_near + x_far) / 2, cy, h + th])
            half = np.array([(x_far - x_near) / 2, hy, th])
            worst = float(
                (_point_box_sdf_np(scan_flat, center, half, 0.0)
                 - scan_radii_flat).min()
            )
            if worst >= -tcfg.max_ref_table_pen:
                break
            x_near += (-tcfg.max_ref_table_pen - worst) + 0.005
        return x_near

    # ascending scan: the lowest slab (tightest duck) whose pushed near
    # edge still leaves the hand min_cover under the table
    h_under, x_near, best = None, None, None
    for h in np.arange(h_base, h_hi + 1e-9, 0.03):
        xn = push_near_edge(float(h), x_near_seed, hy)
        if best is None or xn < best[1]:
            best = (float(h), xn)
        if xn <= pick_reach_x - tcfg.min_cover:
            h_under, x_near = float(h), xn
            break
    if h_under is None:
        h_under, x_near = best
        warnings.append(
            f"near edge x={x_near:.3f} leaves < min_cover "
            f"({tcfg.min_cover}m) of the pick reach (x={pick_reach_x:.3f}) "
            f"under the table"
        )
    x_far = max(x_far, x_near + tcfg.min_depth)

    # Legs: widen the table until the reference clears them.
    leg_centers, leg_half = _make_legs(x_near, x_far, cy, hy, h_under, tcfg)

    def worst_leg_clearance(centers) -> float:
        return min(
            float((_point_box_sdf_np(flat, c, leg_half, 0.0) - radii_flat).min())
            for c in centers
        )

    while worst_leg_clearance(leg_centers) < tcfg.leg_clearance \
            and hy < tcfg.max_half_width:
        hy = min(hy + tcfg.widen_step, tcfg.max_half_width)
        leg_centers, leg_half = _make_legs(x_near, x_far, cy, hy, h_under, tcfg)

    # Footprint may have grown: re-verify the slab once at the final width.
    x_near2 = push_near_edge(h_under, x_near, hy)
    if x_near2 > x_near + 1e-6:
        x_near = x_near2
        x_far = max(x_far, x_near + tcfg.min_depth)
        leg_centers, leg_half = _make_legs(x_near, x_far, cy, hy, h_under, tcfg)

    # snap depth/width UP to the size grid, for a discrete slab-size
    # family downstream. Both directions are safe: the binding near edge is
    # untouched, the far edge extends past the pick reach, and widening is
    # the leg-clearance loop's own direction.
    if tcfg.size_quantum:
        q = tcfg.size_quantum
        x_far = x_near + float(np.ceil((x_far - x_near - 1e-9) / q)) * q
        hy = float(np.ceil((2 * hy - 1e-9) / q)) * q / 2
        leg_centers, leg_half = _make_legs(x_near, x_far, cy, hy, h_under, tcfg)

    leg_clear = worst_leg_clearance(leg_centers)
    if leg_clear < tcfg.leg_clearance:
        warnings.append(
            f"leg clearance {leg_clear:.3f}m < {tcfg.leg_clearance}m "
            f"at max width {2 * hy:.2f}m"
        )

    th = tcfg.slab_thickness / 2
    spec = TableSpec(
        slab_center=_rot_xy(np.array([(x_near + x_far) / 2, cy, h_under + th]),
                            psi),
        slab_half=np.array([(x_far - x_near) / 2, hy, th]),
        leg_centers=_rot_xy(leg_centers, psi),
        leg_half=leg_half,
        yaw=psi,
    )
    if tcfg.underside_drop:
        h_under = _shift_height_in_place(spec, -tcfg.underside_drop)

    # how deep the reference head goes into the slab plane during the deep
    # phase. Positive = the solve must crouch below the reference; ~zero or
    # negative = the table only grazes and there is nothing to avoid.
    head_conflict = float(head_top[deep].max() - spec.underside_z)
    # the head may legitimately stay partly behind the near edge (leaning,
    # arms reaching under); only one that NEVER makes it under is a loss
    head_reach = float(head[deep, 0].max()) + head_r
    if x_near > head_reach + 1e-6:
        warnings.append(
            f"near edge x={x_near:.3f} pushed past the whole deep-phase "
            f"head sweep (reach x={head_reach:.3f}) — no head-under-table "
            f"interaction; the entry sweep conflicts too much with any "
            f"closer slab")

    # World-frame check against the yawed boxes (same SDF path as the reward)
    ref_min_clearance = float(
        clearance_profile_prims(points_w, radii, spec.primitives()).min()
    )
    spec.meta = {
        "h_under": h_under,
        "x_near": x_near,
        "x_far": x_far,
        "H_stand": H_stand,
        "H_min": H_min,
        "pick_reach_x": pick_reach_x,
        "deep_frames": (int(np.flatnonzero(deep)[0]),
                        int(np.flatnonzero(deep)[-1]), int(deep.sum())),
        "duck_frames": (int(np.flatnonzero(duck)[0]),
                        int(np.flatnonzero(duck)[-1]), int(duck.sum())),
        "yaw": psi,
        "ref_min_clearance": ref_min_clearance,
        "head_conflict": head_conflict,
        "worst_leg_clearance": leg_clear,
        "warnings": warnings,
    }

    if verbose:
        d0, d1, dn = spec.meta["deep_frames"]
        print(f"  table: underside z={h_under:.3f} (head stand {H_stand:.2f}, "
              f"min {H_min:.2f}), slab x=[{x_near:.3f}, {x_far:.3f}], "
              f"y={cy:.3f}±{hy:.2f}, yaw={np.degrees(psi):+.1f}deg")
        print(f"  deep phase frames {d0}-{d1} ({dn}), pick reach "
              f"x={pick_reach_x:.3f}, head conflict {head_conflict:+.3f}m, "
              f"ref clearance {ref_min_clearance:.3f}m, "
              f"leg clearance {leg_clear:.3f}m")
        for w in warnings:
            print(f"  WARNING: {w}")
    return spec


# ------------------------------------------------------ task preservation --

def pick_hand_error(
    aug_qpos: np.ndarray, ref_qpos: np.ndarray, mj_model: mujoco.MjModel
) -> float:
    """Distance (m) between augmented and reference pick hand at the pick
    frame (deepest reference reach).

    The task-preservation check: under an aggressively low table the solve
    can go collision-free by simply not reaching under it, which the
    penetration and root-drift checks cannot see.
    """
    n = min(len(aug_qpos), len(ref_qpos))
    if n == 0:
        return 0.0
    hands_ref = _palm_trajectory(ref_qpos[:n], mj_model)
    hands_aug = _palm_trajectory(aug_qpos[:n], mj_model)
    pick = int(np.argmax(hands_ref[:, :, 0].max(axis=1)))
    hand = int(np.argmax(hands_ref[pick, :, 0]))
    return float(np.linalg.norm(hands_aug[pick, hand] - hands_ref[pick, hand]))


def root_pose_errors(aug_qpos: np.ndarray,
                     ref_qpos: np.ndarray) -> Tuple[float, float]:
    """Max pelvis position (m) / orientation (rad) drift vs the matching
    interpolated reference."""
    n = min(len(aug_qpos), len(ref_qpos))
    if n == 0:
        return 0.0, 0.0
    pos_err = float(
        np.linalg.norm(aug_qpos[:n, :3] - ref_qpos[:n, :3], axis=1).max())
    dots = np.abs((aug_qpos[:n, 3:7] * ref_qpos[:n, 3:7]).sum(axis=1))
    rot_err = float((2 * np.arccos(np.clip(dots, -1.0, 1.0))).max())
    return pos_err, rot_err


# -------------------------------------------------------------- injection --

def inject_obstacle_boxes(
    root: ET.Element,
    boxes: list,
    contact_pair_margin: float = 0.04,
    hard_collision: bool = True,
    body_name: str = "obstacles",
) -> None:
    """Add named obstacle boxes + contact pairs into a robot_scene_tree()
    root. ``boxes`` = [(geom_name, center(3), half(3), yaw, rgba)].

    hard_collision=True: the pairs enter the solver and physics blocks
    penetration. False: a huge ``gap`` leaves contacts detected but
    forceless (ghost obstacles), so avoidance comes from the SDF reward.
    """
    worldbody = root.find("worldbody")
    contact = root.find("contact")

    # bump max_contact_points / max_geom_pairs for the added pairs
    custom = root.find("custom")
    for num in custom.findall("numeric"):
        if num.get("name") in ("max_contact_points", "max_geom_pairs"):
            old = int(num.get("data", "15"))
            num.set("data",
                    str(old + len(boxes) * len(ROBOT_COLLISION_GEOMS)))

    body = ET.SubElement(worldbody, "body",
                         {"name": body_name, "pos": "0 0 0"})
    for name, center, half, yaw, rgba in boxes:
        geom_attrs = {
            "name": name,
            "type": "box",
            "pos": f"{center[0]:.4f} {center[1]:.4f} {center[2]:.4f}",
            "size": f"{half[0]:.4f} {half[1]:.4f} {half[2]:.4f}",
            "rgba": rgba,
            "contype": "0",
            "conaffinity": "0",
        }
        if abs(yaw) > 1e-6:
            c, s = np.cos(yaw / 2), np.sin(yaw / 2)
            geom_attrs["quat"] = f"{c:.6f} 0 0 {s:.6f}"
        ET.SubElement(body, "geom", geom_attrs)

    # mujoco_warp's NATIVECCD box-box collider rejects nonzero margins, so
    # pairs against a box robot geom (the BrainCo palms) go margin-free and
    # first touch at the surface; the SDF reward still keeps clearance
    box_robot_geoms = {g.get("name") for g in root.iter("geom")
                       if g.get("type") == "box"}

    for obs_geom, _, _, _, _ in boxes:
        for robot_geom in ROBOT_COLLISION_GEOMS:
            pair_attrs = {
                "name": f"{obs_geom}_{robot_geom}",
                "geom1": robot_geom,
                "geom2": obs_geom,
                # 20 ms: compliant enough that limb-obstacle contact
                # nudges rather than shoves (0.008 tripped the robot)
                "solref": "0.02 1",
                "friction": "1 1",
                "condim": "3",
                "margin": "0" if robot_geom in box_robot_geoms
                else f"{contact_pair_margin:.4f}",
            }
            if not hard_collision:
                # includemargin = margin - gap < 0 always: recorded in
                # the contact array, never entering the solver
                pair_attrs["gap"] = "1.0"
            ET.SubElement(contact, "pair", pair_attrs)


def table_boxes(table: TableSpec) -> list:
    """The legged table as inject_obstacle_boxes input: slab + 4 legs."""
    boxes = [("table_geom", table.slab_center, table.slab_half, table.yaw,
              "0.55 0.35 0.2 1.0")]
    for i, c in enumerate(table.leg_centers):
        boxes.append((f"table_leg_{i}_geom", c, table.leg_half, table.yaw,
                      "0.45 0.28 0.15 1.0"))
    return boxes


# --------------------------------------------------------------- emission --

def emit_obstacle_trial(
    qpos: np.ndarray,
    meta: Dict,
    boxes: list,
    primitives: list,
    geom_names: list,
    out_root: Path,
    task: str,
    params: Dict,
    task_type: str,
    info_extra: Dict,
    data_id: int = 0,
) -> Path:
    """Write a complete obstacle-task trial (shared by under_table / kick):

        <out_root>/processed/kimodo/unitree_g1/humanoid/<task>/
            scene.xml                     robot + injected obstacle boxes
            task_info.json                scene primitives + placement meta
            0/trajectory_kinematic.npz    qpos/qvel/ctrl reference

    Same layout family as the box-carry trials, under the ``humanoid``
    embodiment: obstacles are static terrain, not tracked objects.
    """
    root = robot_scene_tree()
    inject_obstacle_boxes(
        root, boxes,
        contact_pair_margin=float(params["contact_pair_margin"]),
        hard_collision=bool(params["hard_collision"]))
    scene_xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(scene_xml)
    layout.check_robot(model)

    task_dir = (Path(out_root) / "processed" / "kimodo" / "unitree_g1"
                / "humanoid" / task)
    trial_dir = task_dir / str(data_id)
    trial_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "scene.xml").write_text(scene_xml)
    # a re-estimate invalidates any previous solve of the old scene
    stale = trial_dir / "trajectory_aug.npz"  # solve.mppi_loop.RESULT_NPZ
    if stale.exists():
        stale.unlink()

    fps = meta["fps"]
    # trial files use the model layout; hand joints ride along at rest
    qpos = layout.to_model(model, qpos)
    qvel = compute_qvel(model, qpos, 1.0 / fps)
    ctrl = layout.ctrl_reference(model, qpos)
    np.savez(trial_dir / "trajectory_kinematic.npz",
             qpos=qpos, qvel=qvel, ctrl=ctrl)

    info = {
        "task_type": task_type,
        "ref_dt": 1.0 / fps,
        "source_npz": meta["file_path"],
        "scene_params": {k: params[k] for k in sorted(params)},
        "primitives": primitives,
        "obstacle_geoms": geom_names,
        # the pair set, recorded so the solve needs no recon-side import
        "robot_collision_geoms": ROBOT_COLLISION_GEOMS,
        **info_extra,
    }
    (task_dir / "task_info.json").write_text(json.dumps(info, indent=2))
    return trial_dir


def emit_trial(
    qpos: np.ndarray,
    meta: Dict,
    spec: TableSpec,
    out_root: Path,
    task: str,
    params: Dict,
    data_id: int = 0,
) -> Path:
    """Write a complete under-table trial (see emit_obstacle_trial)."""
    return emit_obstacle_trial(
        qpos, meta, table_boxes(spec), spec.primitives(), spec.geom_names(),
        out_root, task, params, "under_table",
        {
            "table_yaw": spec.yaw,
            "table_underside_z": spec.underside_z,
            **{k: spec.meta[k] for k in
               ("h_under", "x_near", "x_far", "H_stand", "H_min",
                "pick_reach_x", "deep_frames", "duck_frames",
                "ref_min_clearance", "head_conflict",
                "worst_leg_clearance", "warnings")},
        },
        data_id=data_id)


def reconstruct(
    npz_path: Path,
    out_root: Path,
    task: str,
    params: Dict,
) -> Tuple[Optional[Path], Optional[TableSpec], str]:
    """Full under-table reconstruction of one clip with resolved ``params``
    -> (task_dir or None, spec or None, one-line summary). None means no
    duck phase was found, a verdict rather than an error."""
    from .loader import load_kimodo_npz  # local: keeps import cycles short

    qpos, meta = load_kimodo_npz(npz_path)  # smooth root: duck-in-place clip
    model = robot_model()
    rng = np.random.RandomState(int(params["scene_seed"]))
    try:
        spec = estimate_table(qpos, model, SimpleNamespace(**params), rng=rng)
    except ValueError as e:
        return None, None, f"{task}: reconstruction SKIPPED ({e})"

    trial = emit_trial(qpos, meta, spec, Path(out_root), task, params)
    m = spec.meta
    line = (f"{task}: table underside z={m['h_under']:.3f} "
            f"x=[{m['x_near']:.2f},{m['x_far']:.2f}] "
            f"yaw {np.degrees(spec.yaw):+.0f}deg seed {params['scene_seed']} "
            f"head-conf {m['head_conflict']:+.3f} "
            f"ref-clr {m['ref_min_clearance']:+.3f} "
            f"warn [{'; '.join(m['warnings']) or '-'}] -> {trial}")
    return trial.parent, spec, line
