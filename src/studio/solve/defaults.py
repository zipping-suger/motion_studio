"""The SBMPC solve's tunable hyperparameters: one flat dict every task
shares, plus which of them must stay integers.

The solver's only parameter contract: `spider_cfg.build` layers a fully
resolved dict of these on SPIDER's fixed task setup, and `coerce`
validates keys against them. Task-specific values (a pole's grip
weighting, the chair's grasp terms) are the TASK's business —
`studio.tasks.Task.solve_params` resolves them before the solve
subprocess is launched, so nothing here ever needs a task name. The
values are also what `studio.config` exposes as the box_carry defaults
config.yml's top-level ``solve:`` section overrides.
"""

# object interaction is the objective, robot tracking only guides. The
# fixed task setup lives in spider_cfg.FIXED.
SOLVE_DEFAULTS = {
    "num_samples": 2048,        # parallel mujoco_warp worlds (VRAM-bound)
    "horizon": 1.0,             # s, planning horizon
    "knot_dt": 0.1,             # s, spline knot spacing: how fast sampled
                                # exploration noise may turn. Snapped to a
                                # whole number of sim steps that tiles the
                                # horizon (solve.spider_cfg.snap_knot_dt).
    "max_num_iterations": 16,   # annealing iterations per control tick
    "temperature": 1.0,         # softmax temperature over the top 10%
    "joint_rew_scale": 0.5,     # robot joints (guide only)
    "base_pos_rew_scale": 1.0,
    "base_rot_rew_scale": 1.0,
    "pos_rew_scale": 8.0,       # object position (the objective)
    "rot_rew_scale": 0.2,       # object orientation (barely matters for a box)
    "contact_rew_scale": 2.0,   # pulls each holding palm onto the object
                                # through its hold window, so intermittent
                                # slap-and-release prices worse than a
                                # maintained hold. box_carry uses the
                                # simulated box's grasp faces, the others
                                # the baked palm track.
    "vel_rew_scale": 1.0,       # master scale of the per-block velocity term
    "grip_rew_scale": 1.0,      # keep the SIMULATED object in the palm's
                                # calibrated pocket: slip shows as
                                # palm-relative drift long before a drop
    "upright_rew_scale": 10.0,  # anti-fall: penalty for tilting PAST the
                                # reference's own tilt or sinking BELOW
                                # its height, so bending clips stay free.
                                # Tracking alone prices a fall as another
                                # pose error, and MPPI takes that trade.
    "obj_end_rew_scale": 1.5,   # object-tracking multiplier at the
                                # interaction's endpoints, which ARE the
                                # objective...
    "obj_mid_rew_scale": 0.5,   # ...and mid-interaction, where the
                                # carried path matters less
    "self_collision_rew_scale": 50.0,  # penalty per metre of summed
                                # penetration across the template's ghost
                                # self-pairs. They exert no force, so this
                                # term is what steers rollouts apart.
    "grasp_rew_scale": 0.0,     # reward per hand for its geoms (palm +
                                # fingers) touching the object, saturating
                                # at 6 contacts, over its contact window.
                                # Off for the box; the chair turns it on.
    "grasp_pen_rew_scale": 0.0,  # penalty per metre of hand-object
                                # penetration beyond 5 mm, summed over
                                # contacts: a member buried in the palm
                                # box is caged and carried by penetration
                                # force, not held. Off for the box (its
                                # reference squeezes in on purpose).
    "support_rew_scale": 0.0,   # reward per body-support group (torso,
                                # hip, thigh, forearm the reference rests
                                # the object on) for touching it over its
                                # window. Off unless the trial lists
                                # supports (the chair does).
}

# solve params that must stay integers when they reach the optimizer
SOLVE_INT_KEYS = frozenset({"num_samples", "max_num_iterations"})
