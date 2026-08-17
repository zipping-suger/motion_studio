"""Kick task parameters: defaults every layer shares.

Pure data, importable everywhere (no mujoco, no torch) — same contract as
`under_table_params`. The obstacle is a floor-standing box placed in the
kicking foot's approach path: the reference swing penetrates it (that IS
the augmentation signal), the solve re-paths the kick around it, and the
kick point itself stays reachable.

Solve defaults start from the under-table task's certified numbers with
the leg-specific inversions a kick needs (the LEG is the avoiding limb
here, so it must keep its exploration noise and get room to deviate).
They have NOT been difficulty-swept yet — expect to retune.
"""

# --- scene: kick detection + obstacle placement (recon side) -------------
SCENE_DEFAULTS = {
    "scene_seed": 0,           # rng for jitter + yaw; the randomization knob
    "hard_collision": True,    # solid obstacle; SDF reward keeps it grazing
    "contact_pair_margin": 0.04,

    "lift_thresh": 0.10,       # foot rise above stance that counts as swing
    # Obstacle position along the approach: the difficulty knob. Later =
    # closer to the kick = fewer frames to re-converge after the dodge.
    # First real clip (2026-08-14): 0.45 with box_width 0.40 failed — the
    # foot detoured 23cm laterally and never got back (kick_err 0.36);
    # 0.25 with box_width 0.25 passes (kick_err 0.10, clearance 20mm).
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
    # the legs do the avoiding here — keep their exploration noise
    # (under_table damps it to 0.3 because there the arms explore)
    "leg_noise_scale": 1.0,
    "terminate_resample": True,
    "base_pos_threshold": 0.25,
    "base_rot_threshold": 0.3,
    # reward: tracking
    "vel_rew_scale": 0.0001,
    "qw_base_pos": 15.0,
    "qw_base_rot": 10.0,
    # legs get room to deviate around the obstacle (under_table pins them
    # at 10 for balance while the arms deviate); collision-free beats
    # tracking here, so the leg is barely pinned at all
    "qw_legs": 3.0,
    "qw_waist": 5.0,
    "qw_shoulder_elbow": 2.0,
    "qw_wrist": 2.0,           # hand pose is irrelevant to a kick
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
    # collision-free is the objective: PASS needs real clearance, not just
    # bounded penetration
    "min_clearance": 0.01,     # worst-case clearance vs the obstacle (m)
    # drift bounds are fall detectors, not fidelity — relaxed accordingly
    # (knocked-over shows up as >= ~0.5m in the mppi_obstacle sweeps)
    "max_root_pos_err": 0.25,
    "max_root_rot_err": 0.45,
    # task preservation: kicking foot vs the reference kick point at the
    # kick frame — a solve can go collision-free by not kicking at all,
    # and this is the check that catches it. Loose: the kick must happen,
    # not mirror the reference.
    "max_kick_err": 0.25,
}
