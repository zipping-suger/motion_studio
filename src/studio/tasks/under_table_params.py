"""Under-table task parameters: defaults every layer shares.

Pure data, importable everywhere — the CLI (studio venv) validates flags
against these, `studio.recon.table` consumes the scene set, and
`studio.solve.mppi_loop` (solve venv) consumes the solve set — so nothing
here may import mujoco or torch.

Values are the mppi_obstacle experiment's certified defaults
(obstacle_config.yml; see its README for the difficulty sweeps behind
max_ref_table_pen / underside_drop / yaw_range and the reward-shaping
notes). config.yml's ``tasks: under_table:`` section overrides per machine,
``--set key=value`` per run.
"""

# --- scene: table estimation + scene generation (recon side) -------------
SCENE_DEFAULTS = {
    "scene_seed": 0,           # rng for jitter + yaw; the randomization knob
    "hard_collision": True,    # solid table; SDF reward keeps contacts grazing
    "contact_pair_margin": 0.04,
    # Arms may conflict with the slab: placement (underside + near-edge
    # scan) is driven by head/torso/legs only, so a reach at slab height
    # cannot push the table away from the head — re-pathing the arms is
    # the solver's work. False restores whole-body placement scans.
    "arm_conflict": True,

    "slab_thickness": 0.05,
    "table_width": 1.2,        # initial y extent; legs straddle the body sweep
    "max_half_width": 0.9,     # widen limit for the leg-clearance loop
    "size_quantum": 0.01,      # snap slab depth/width UP to this grid (m)

    "duck_frac": 0.75,         # duck window: head_top < frac * standing head
    "deep_band": 0.10,         # deep phase: head_top <= min head_top + band
    "head_clearance": 0.04,    # underside above deep-phase body tops
    "underside_min": 0.65,
    "underside_max": 1.00,
    "min_duck": 0.30,          # underside <= standing head top - min_duck
    "far_margin": 0.15,        # slab extends past the deepest hand reach
    "min_cover": 0.15,         # hand must end >= this past the near edge
    "min_depth": 0.40,         # slab x-depth floor
    "back_start": 0.10,        # near-edge scan seed offset behind the deep head
    "max_ref_table_pen": 0.06,  # allowed reference slab pen (closeness knob)
    "underside_drop": 0.04,    # lower the finished slab (height knob)
    "leg_cross": 0.06,         # leg square cross-section
    "leg_inset": 0.05,         # leg inset from slab edges
    "leg_clearance": 0.04,     # required reference clearance vs legs
    "widen_step": 0.05,
    "h_jitter_down": 0.02,     # per-seed variety (down = small deliberate
    "h_jitter_up": 0.05,       # reference conflict)
    "y_jitter": 0.05,
    "yaw_range": 0.21,         # yaw sampled in +/-0.21 rad (12 deg); 0 = off
}

# --- solve: MPPI sampling + reward weights (solve side) ------------------
SOLVE_DEFAULTS = {
    # sampling
    "horizon": 0.8,            # s, planning horizon
    "num_samples": 2048,       # parallel mujoco_warp worlds (VRAM-bound)
    "max_num_iterations": 16,  # annealing iterations per control tick
    "temperature": 0.6,
    "first_ctrl_noise_scale": 0.5,
    "last_ctrl_noise_scale": 1.0,
    "final_noise_scale": 0.1,
    "joint_noise_scale": 0.1,
    "leg_noise_scale": 0.3,    # damp leg exploration; arms do the avoiding
    "terminate_resample": True,  # kill & replace diverged samples
    "base_pos_threshold": 0.25,
    "base_rot_threshold": 0.3,
    # reward: tracking
    "vel_rew_scale": 0.0001,
    "qw_base_pos": 15.0,       # per-group qpos tracking weights (nv=35)
    "qw_base_rot": 10.0,
    "qw_legs": 10.0,
    "qw_waist": 5.0,
    "qw_shoulder_elbow": 2.0,  # arms free to deviate for avoidance
    "qw_wrist": 10.0,
    # reward: stability + avoidance
    "stability_weight": 50.0,  # tilt term is reference-relative
    "sdf_weight": 20.0,
    "collision_margin": 0.03,
    "collision_inner_margin": 0.015,  # steep inner hinge: keep real clearance
    "collision_inner_scale": 10.0,
    "collision_count_weight": 40.0,
    # reward: drift (quadratic, outside the coupled tracking norm)
    "base_pos_drift_weight": 100.0,
    "base_rot_drift_weight": 50.0,
    "base_vel_weight": 2.0,    # catches a forward trip while still momentum
    "arm_vel_weight": 0.05,    # damps press-shove-press at the table
}

SOLVE_INT_KEYS = frozenset({"num_samples", "max_num_iterations"})
SOLVE_BOOL_KEYS = frozenset({"terminate_resample"})

# --- verify: acceptance thresholds (evaluation, studio venv) -------------
VERIFY_DEFAULTS = {
    # collision-free is the objective: PASS needs real clearance, not just
    # bounded penetration — grazing solutions are untrackable downstream
    # (any tracking error there means solid contact)
    "min_clearance": 0.01,     # worst-case clearance vs any primitive (m)
    "max_root_pos_err": 0.15,  # fall detector, not fidelity (m)
    "max_root_rot_err": 0.30,  # (rad)
    "max_pick_err": 0.15,      # pick hand vs reference pick point (m)
}


def coerce(value, default):
    """Cast a CLI/YAML string to the type its default declares."""
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, int):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    return value
