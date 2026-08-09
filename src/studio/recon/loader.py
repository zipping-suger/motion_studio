"""Kimodo NPZ -> MuJoCo G1 (29 DoF) qpos.

Ported from mppi_locoma's `kimodo_loader.py`; the joint map and the
coordinate convention are unchanged. The one difference: the G1 model
comes from this repo's scene template (see `assets.robot_xml`) rather
than from an installed SPIDER.

Kimodo NPZ format:
    local_rot_mats:  (T, 34, 3, 3) float32 - local rotation matrices (y-up)
    global_rot_mats: (T, 34, 3, 3) float32 - global rotation matrices (y-up)
    posed_joints:    (T, 34, 3)    float32 - joint world positions (y-up)
    root_positions:  (T, 3)        float32 - pelvis world position (y-up)
    smooth_root_pos: (T, 3)        float32 - drift-removed pelvis position
    foot_contacts:   (T, 4)        bool    - [L_heel, L_toe, R_heel, R_toe]
    global_root_heading: (T, 2)    float32

Kimodo 34-joint skeleton:
    j0        : pelvis (root)
    j1  - j6  : left leg   (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    j7        : left toe   (no G1 DOF)
    j8  - j13 : right leg  (same order)
    j14       : right toe  (no G1 DOF)
    j15 - j17 : waist      (yaw, roll, pitch)
    j18 - j24 : left arm   (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
    j25       : left hand tip  (no G1 DOF)
    j26 - j32 : right arm  (same order)
    j33       : right hand tip (no G1 DOF)

Coordinate convention:
    Kimodo: x = lateral,  y = up,      z = forward
    MuJoCo: x = forward,  y = lateral, z = up
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from . import assets

FPS = 30.0

# Kimodo (y-up) -> MuJoCo (z-up):  mj_x = kim_z,  mj_y = kim_x,  mj_z = kim_y
_KIM2MJ = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
_MJ2KIM = _KIM2MJ.T

# Kimodo indices of keypoints that have no G1 DOF
KIM_PELVIS = 0
KIM_LEFT_TOE, KIM_RIGHT_TOE = 7, 14
KIM_LEFT_ANKLE, KIM_RIGHT_ANKLE = 6, 13
KIM_LEFT_HAND_TIP, KIM_RIGHT_HAND_TIP = 25, 33
KIM_LEFT_WRIST, KIM_RIGHT_WRIST = 24, 32

DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# (kimodo_child_idx, g1_child_body_name, revolute_axis_mj) per DOF, in DOF_NAMES order
_JOINT_MAP: List[Tuple[int, str, List[int]]] = [
    # left leg
    (1, "left_hip_pitch_link", [0, 1, 0]),
    (2, "left_hip_roll_link", [1, 0, 0]),
    (3, "left_hip_yaw_link", [0, 0, 1]),
    (4, "left_knee_link", [0, 1, 0]),
    (5, "left_ankle_pitch_link", [0, 1, 0]),
    (6, "left_ankle_roll_link", [1, 0, 0]),
    # right leg
    (8, "right_hip_pitch_link", [0, 1, 0]),
    (9, "right_hip_roll_link", [1, 0, 0]),
    (10, "right_hip_yaw_link", [0, 0, 1]),
    (11, "right_knee_link", [0, 1, 0]),
    (12, "right_ankle_pitch_link", [0, 1, 0]),
    (13, "right_ankle_roll_link", [1, 0, 0]),
    # waist
    (15, "waist_yaw_link", [0, 0, 1]),
    (16, "waist_roll_link", [1, 0, 0]),
    (17, "torso_link", [0, 1, 0]),
    # left arm
    (18, "left_shoulder_pitch_link", [0, 1, 0]),
    (19, "left_shoulder_roll_link", [1, 0, 0]),
    (20, "left_shoulder_yaw_link", [0, 0, 1]),
    (21, "left_elbow_link", [0, 1, 0]),
    (22, "left_wrist_roll_link", [1, 0, 0]),
    (23, "left_wrist_pitch_link", [0, 1, 0]),
    (24, "left_wrist_yaw_link", [0, 0, 1]),
    # right arm
    (26, "right_shoulder_pitch_link", [0, 1, 0]),
    (27, "right_shoulder_roll_link", [1, 0, 0]),
    (28, "right_shoulder_yaw_link", [0, 0, 1]),
    (29, "right_elbow_link", [0, 1, 0]),
    (30, "right_wrist_roll_link", [1, 0, 0]),
    (31, "right_wrist_pitch_link", [0, 1, 0]),
    (32, "right_wrist_yaw_link", [0, 0, 1]),
]


def _rotmat_kim_to_mj(R: np.ndarray) -> np.ndarray:
    return _KIM2MJ @ R @ _MJ2KIM


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z), Shepperd method."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        return np.array([0.25 * s,
                         (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                         (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def robot_model() -> mujoco.MjModel:
    """The G1 model used for FK and joint extraction."""
    return mujoco.MjModel.from_xml_string(assets.robot_xml())


def _build_rest_rotations(model: mujoco.MjModel) -> Dict[str, np.ndarray]:
    """{body_name: R_body_in_parent} from zero-angle FK.

    Some G1 bodies (shoulder/hip chain) have non-identity body-frame offsets
    baked into the XML quat attributes; they must be removed before extracting
    joint angles.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rest = {}
    for bid in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        pid = model.body_parentid[bid]
        R_p = data.xmat[pid].reshape(3, 3)
        R_c = data.xmat[bid].reshape(3, 3)
        rest[name] = R_p.T @ R_c
    return rest


def _extract_revolute_angle(R_joint: np.ndarray, axis: np.ndarray) -> float:
    skew = np.array([R_joint[2, 1] - R_joint[1, 2],
                     R_joint[0, 2] - R_joint[2, 0],
                     R_joint[1, 0] - R_joint[0, 1]])
    return float(np.arctan2(np.dot(axis, skew) / 2.0,
                            (np.trace(R_joint) - 1.0) / 2.0))


def compute_qvel(model: mujoco.MjModel, qpos: np.ndarray,
                 dt: float) -> np.ndarray:
    """Finite-difference qvel (T, nv) handling quaternions via MuJoCo."""
    T = len(qpos)
    qvel = np.zeros((T, model.nv))
    for t in range(T - 1):
        mujoco.mj_differentiatePos(model, qvel[t], dt, qpos[t], qpos[t + 1])
    if T > 1:
        qvel[-1] = qvel[-2]
    return qvel


def load_kimodo_npz(
    filepath: Path,
    robot_xml: Optional[Path] = None,
    use_smooth_pos: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """Load a Kimodo NPZ and convert to MuJoCo G1 qpos.

    Returns:
        qpos:     (T, 36) float64 — pelvis pos(3) + quat wxyz(4) + 29 joints.
        metadata: dict with fps, n_frames, dof_names, foot_contacts (T,4),
                  joint_positions (T,34,3) in MuJoCo coords,
                  global_root_heading, file_path.
    """
    filepath = Path(filepath)
    model = (mujoco.MjModel.from_xml_path(str(robot_xml)) if robot_xml
             else robot_model())

    d = np.load(filepath, allow_pickle=True)
    local_rots = d["local_rot_mats"].astype(np.float64)    # (T, 34, 3, 3)
    global_rots = d["global_rot_mats"].astype(np.float64)  # (T, 34, 3, 3)
    posed_j = d["posed_joints"].astype(np.float64)         # (T, 34, 3) y-up
    foot_contacts = d["foot_contacts"]                     # (T, 4) bool

    if use_smooth_pos and "smooth_root_pos" in d:
        root_pos = d["smooth_root_pos"].astype(np.float64)
    else:
        root_pos = d["root_positions"].astype(np.float64)

    T = local_rots.shape[0]

    rest = _build_rest_rotations(model)
    axes = [np.array(ax, dtype=np.float64) for _, _, ax in _JOINT_MAP]
    corrs = [rest[name] for _, name, _ in _JOINT_MAP]

    qpos = np.zeros((T, 36), dtype=np.float64)
    for t in range(T):
        p = root_pos[t]
        qpos[t, 0] = p[2]   # forward
        qpos[t, 1] = p[0]   # lateral
        qpos[t, 2] = p[1]   # up
        qpos[t, 3:7] = _rotmat_to_quat_wxyz(_rotmat_kim_to_mj(global_rots[t, 0]))
        for qi, (kim_idx, _, _) in enumerate(_JOINT_MAP):
            R_local_mj = _rotmat_kim_to_mj(local_rots[t, kim_idx])
            R_joint = corrs[qi].T @ R_local_mj
            qpos[t, 7 + qi] = _extract_revolute_angle(R_joint, axes[qi])

    joint_positions_mj = np.stack(
        [posed_j[..., 2], posed_j[..., 0], posed_j[..., 1]], axis=-1
    )

    metadata: Dict = {
        "fps": FPS,
        "n_frames": T,
        "file_path": str(filepath),
        "dof_names": DOF_NAMES,
        "foot_contacts": foot_contacts,
        "joint_positions": joint_positions_mj,
        "global_root_heading": d.get("global_root_heading"),
    }
    return qpos, metadata


def load_kimodo_constraints(npz_path: Path) -> Dict:
    """Load batch_inputs constraints.json + meta.json paired with an NPZ.

    Returns dict with pick_frame, active_hands, text, raw — or {} if absent.
    For Table_pick batches the hand constraints may be missing; active hands
    are then inferred from the meta text and pick_frame from the root2d
    constraint's last frame index.
    """
    npz_path = Path(npz_path)
    motion_id = npz_path.parent.name
    batch_root = npz_path.parent.parent
    con_path = batch_root / "batch_inputs" / motion_id / "constraints.json"
    meta_path = batch_root / "batch_inputs" / motion_id / "meta.json"
    if not con_path.exists():
        return {}

    cons: List[Dict] = json.loads(con_path.read_text())
    text = json.loads(meta_path.read_text()).get("text", "") \
        if meta_path.exists() else ""

    hand_cons = [c for c in cons if c["type"] in ("right-hand", "left-hand")]
    active_hands: List[str] = []
    pick_frame = None
    if hand_cons:
        pick_frame = hand_cons[0]["frame_indices"][-1]
        for c in hand_cons:
            active_hands.append("right" if "right" in c["type"] else "left")
    if not active_hands and text:
        tl = text.lower()
        if "right hand" in tl:
            active_hands.append("right")
        if "left hand" in tl:
            active_hands.append("left")
        if not active_hands and ("both hands" in tl or "object" in tl):
            active_hands = ["left", "right"]
    if pick_frame is None:
        root_cons = [c for c in cons if c["type"] == "root2d"]
        pick_frame = root_cons[0]["frame_indices"][-1] if root_cons else 0

    return {
        "pick_frame": pick_frame,
        "active_hands": active_hands,
        "text": text,
        "raw": cons,
    }
