"""Build SPIDER's Config directly — no hydra.

FIXED is the whole fixed task setup as one dict; the hyperparameters
worth tuning live in `defaults.SOLVE_DEFAULTS` and layer on top per run,
already resolved by the launcher (a task's own values arrive as explicit
params, never by task name). Values are validated literals — `sim_dt` is
0.0166667 rather than 1/60 because it becomes the MuJoCo timestep, so
the rounding reproduces.
"""

from __future__ import annotations

from pathlib import Path

from spider.config import Config, process_config

from .defaults import SOLVE_DEFAULTS, SOLVE_INT_KEYS

FIXED = {
    # --- task -----------------------------------------------------------
    "robot_type": "unitree_g1",
    "embodiment_type": "humanoid_object",
    "dataset_name": "kimodo",
    "data_id": 0,
    "seed": 0,
    # --- simulator ------------------------------------------------------
    "simulator": "mjwp",
    "device": "cuda:0",
    "sim_dt": 0.0166667,      # 1/60, as validated; becomes the mj timestep
    "ctrl_dt": 0.1,           # one control tick = 6 sim steps
    "ref_dt": 0.0333333,      # 1/30; task_info.json overrides this anyway
    "render_dt": 0.02,
    "max_sim_steps": -1,      # -1: derive from the reference length
    "nconmax_per_env": 120,
    "njmax_per_env": 350,
    "num_dyn": 1,
    "num_dr": 1,              # one parameter set: margins applied, not varied
    "pair_margin_range": (-0.005, 0.005),
    "xy_offset_range": (-0.005, 0.005),
    "perturb_force": 0.0,
    "perturb_torque": 0.0,
    "contact_guidance": False,
    "gibbs_sampling": False,
    # --- optimizer ------------------------------------------------------
    "improvement_threshold": 0.01,
    "improvement_check_steps": 1,
    "first_ctrl_noise_scale": 0.5,
    "last_ctrl_noise_scale": 1.0,
    "final_noise_scale": 0.1,   # beta_traj = this ** (1/max_num_iterations)
    "exploit_ratio": 0.01,
    "exploit_noise_scale": 0.01,
    "joint_noise_scale": 0.1,
    "pos_noise_scale": 0.03,
    "rot_noise_scale": 0.03,
    "terminal_rew_scale": 1.0,
    "terminate_resample": False,
    "object_pos_threshold": 0.1,
    "object_rot_threshold": 0.3,
    # --- no viewer, no video: studio always solves headless --------------
    "show_viewer": False,
    "viewer": "",
    "rerun_spawn": False,
    "save_video": False,
    "save_rerun": False,
    "save_viser": False,
    "save_metrics": True,
    "save_info": True,          # writes trajectory_mjwp.npz
    "save_config": False,       # studio writes its own solve_config.json
    "wait_on_finish": False,
    "sanity_check_seconds": 0.0,
    # --- traces ---------------------------------------------------------
    "trace_dt": 0.0333333,
    "num_trace_uniform_samples": 4,
    "num_trace_topk_samples": 2,
}


def snap_knot_dt(knot_dt: float, horizon: float, sim_dt: float) -> float:
    """The nearest legal knot spacing to `knot_dt`.

    Knots must land on sim steps AND tile the horizon exactly: SPIDER
    upsamples its (num_samples, num_knots, nu) noise by knot_steps and
    adds the result to a horizon_steps-long control buffer, so
    knot_steps has to divide horizon_steps. A GUI value that misses (any
    knot_dt that is not horizon / an integer) snaps to the nearest one
    that lands, ties going to the finer spacing.
    """
    horizon_steps = int(round(horizon / sim_dt))
    want = max(1, int(round(knot_dt / sim_dt)))
    steps = min((d for d in range(1, horizon_steps + 1)
                 if horizon_steps % d == 0),
                key=lambda d: (abs(d - want), d))
    return steps * sim_dt


def coerce(params: dict | None) -> dict:
    """Cast GUI/CLI values to the types the optimizer needs."""
    out = {}
    for key, val in (params or {}).items():
        if key not in SOLVE_DEFAULTS:
            raise SystemExit(f"unknown solve parameter: {key}")
        # CLI values arrive as strings, GUI ones as floats even for ints
        out[key] = int(float(val)) if key in SOLVE_INT_KEYS else float(val)
    return out


# solve params of studio's own reward terms — SPIDER's Config dataclass
# rejects unknown kwargs, so these ride as attributes set after build
STUDIO_KEYS = ("grip_rew_scale", "grasp_rew_scale", "grasp_pen_rew_scale",
               "support_rew_scale", "self_collision_rew_scale",
               "upright_rew_scale", "obj_end_rew_scale", "obj_mid_rew_scale")


def build(task: str, dataset_dir: Path, params: dict | None = None) -> Config:
    """The fully resolved Config for one trial, ready to run."""
    values = {
        **FIXED,
        **SOLVE_DEFAULTS,
        **coerce(params),
        "task": task,
        "dataset_dir": str(dataset_dir),
    }
    snapped = snap_knot_dt(values["knot_dt"], values["horizon"],
                           values["sim_dt"])
    # half a sim step of slack: sim_dt is a rounded literal, so an
    # already-legal knot_dt comes back a few 1e-7 off
    if abs(snapped - values["knot_dt"]) > values["sim_dt"] / 2:
        print(f"knot_dt {values['knot_dt']:g} does not tile the "
              f"{values['horizon']:g} s horizon in whole sim steps — "
              f"using {snapped:g}")
    values["knot_dt"] = snapped

    extra = {k: values.pop(k) for k in STUDIO_KEYS if k in values}
    config = process_config(Config(**values))
    for key, val in extra.items():
        setattr(config, key, val)

    # loop.py implements exactly this envelope — anything else would take
    # a code path the port dropped
    assert config.embodiment_type == "humanoid_object", config.embodiment_type
    assert not config.contact_guidance, "contact_guidance is not supported"
    assert not config.gibbs_sampling, "gibbs_sampling is not supported"
    assert not config.show_viewer, "the solve is headless"
    return config


def summarize(config: Config) -> dict:
    """JSON-safe view of a resolved config, for per-run provenance."""
    skip = {"noise_scale", "env_params_list", "viewer_body_entity_and_ids"}
    out = {}
    for key, val in vars(config).items():
        if key in skip:
            continue
        out[key] = list(val) if isinstance(val, tuple) else val
    return out
