"""Kick task parameters: defaults every layer shares.

Pure data, importable everywhere (no mujoco, no torch) — same contract as
`under_table_params`. The obstacle is a floor-standing box in the kicking
foot's approach path: the reference swing penetrates it (that IS the
augmentation signal), the solve re-paths around it, and the kick point
stays reachable.

Solve defaults start from under_table's with the inversions a kick needs
— the LEG is the avoiding limb here. Not difficulty-swept yet.
"""

# --- scene: kick detection + obstacle placement (recon side) -------------
SCENE_DEFAULTS = {
    "scene_seed": 0,           # rng for jitter + yaw; the randomization knob
    "hard_collision": True,    # solid obstacle; SDF reward keeps it grazing
    "contact_pair_margin": 0.04,

    "lift_thresh": 0.10,       # foot rise above stance that counts as swing
    # position along the approach: the difficulty knob. Later = closer to
    # the kick = fewer frames to re-converge. 0.45 with box_width 0.40
    # failed (foot detoured 23 cm and never got back, kick_err 0.36);
    # 0.25 with box_width 0.25 passes (kick_err 0.10, clearance 20 mm).
    "path_frac": 0.25,
    "frac_jitter": 0.15,       # per-seed jitter on path_frac
    "y_jitter": 0.06,          # cross-path center jitter (m)
    "yaw_jitter": 0.25,        # obstacle yaw jitter about the path dir (rad)
    "max_ref_pen": 0.05,       # closeness: reference foot sinks this far
                               # below the box top at the chosen path point
    "box_depth": 0.25,         # obstacle footprint along the path (m)
    "box_width": 0.25,         # across the path (m); wider = longer detour
    "min_box_height": 0.15,    # never emit a shorter box (kick too low)
    "body_clearance": 0.05,    # non-kicking body must clear the box
    "clear_step": 0.05,        # cross-path shift step for the clearing loop
    "kick_margin": 0.05,       # kick point must stay outside the box
}

# --- solve: MPPI sampling + reward weights (solve side) ------------------
SOLVE_DEFAULTS = {
    # sampling
    "horizon": 0.8,
    "num_samples": 2048,
    "max_num_iterations": 16,
    "temperature": 0.6,
    "first_ctrl_noise_scale": 0.5,
    "last_ctrl_noise_scale": 1.0,
    "final_noise_scale": 0.1,
    "joint_noise_scale": 0.1,
    # the legs do the avoiding here, so they keep their exploration noise
    "leg_noise_scale": 1.0,
    "hand_noise_scale": 0.0,   # BrainCo fingers (if present) hold rest
    "terminate_resample": True,
    "base_pos_threshold": 0.25,
    "base_rot_threshold": 0.3,
    # reward: tracking
    "vel_rew_scale": 0.0001,
    "qw_base_pos": 15.0,
    "qw_base_rot": 10.0,
    # collision-free beats tracking here, so the leg is barely pinned
    # (under_table holds legs at 10 for balance while the arms deviate)
    "qw_legs": 3.0,
    "qw_waist": 5.0,
    "qw_shoulder_elbow": 2.0,
    "qw_wrist": 2.0,           # hand pose is irrelevant to a kick
    "qw_hands": 0.5,           # BrainCo fingers: rest pose, barely weighted
    # reward: stability + avoidance — avoidance deliberately heavy
    "stability_weight": 50.0,
    "sdf_weight": 30.0,
    "collision_margin": 0.03,
    "collision_inner_margin": 0.015,
    "collision_inner_scale": 10.0,
    "collision_count_weight": 80.0,
    # reward: drift
    "base_pos_drift_weight": 100.0,
    "base_rot_drift_weight": 50.0,
    "base_vel_weight": 2.0,
    "arm_vel_weight": 0.05,
}

SOLVE_INT_KEYS = frozenset({"num_samples", "max_num_iterations"})
SOLVE_BOOL_KEYS = frozenset({"terminate_resample"})

# --- verify: acceptance thresholds (evaluation, studio venv) -------------
VERIFY_DEFAULTS = {
    # PASS needs real clearance, not bounded penetration
    "min_clearance": 0.01,     # worst-case clearance vs the obstacle (m)
    # fall detectors, not fidelity: knocked-over reads >= ~0.5 m
    "max_root_pos_err": 0.25,
    "max_root_rot_err": 0.45,
    # task preservation: a solve can go collision-free by not kicking at
    # all. Loose — the kick must happen, not mirror the reference.
    "max_kick_err": 0.25,
}
