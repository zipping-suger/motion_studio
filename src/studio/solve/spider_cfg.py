"""Build SPIDER's Config directly — no hydra.

FIXED below is the whole solve configuration, resolved, as one dict. It is the
task setup — timesteps, simulator sizing, noise schedule, what is saved.
The hyperparameters worth tuning live in `config.SOLVE_DEFAULTS` and are
layered on top per run.

Values are validated literals, including `sim_dt` as
0.0166667 rather than 1/60: it becomes the MuJoCo timestep
(`spider/simulators/mjwp.py:80`), so the rounding is load-bearing for
reproducing a solve.
"""

from __future__ import annotations

from pathlib import Path

from spider.config import Config, process_config

from ..config import SOLVE_DEFAULTS, SOLVE_INT_KEYS

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
    "knot_dt": 0.1,           # 10 spline knots over a 1.0 s horizon
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


def coerce(params: dict | None) -> dict:
    """Cast GUI/CLI values to the types the optimizer needs."""
    out = {}
    for key, val in (params or {}).items():
        if key not in SOLVE_DEFAULTS:
            raise SystemExit(f"unknown solve parameter: {key}")
        # via the CLI these arrive as strings, and GUI number widgets hand
        # back floats even for the integer params
        out[key] = int(float(val)) if key in SOLVE_INT_KEYS else float(val)
    return out


def build(task: str, dataset_dir: Path, params: dict | None = None) -> Config:
    """The fully resolved Config for one trial, ready to run."""
    values = {
        **FIXED,
        **SOLVE_DEFAULTS,
        **coerce(params),
        "task": task,
        "dataset_dir": str(dataset_dir),
    }
    config = process_config(Config(**values))

    # studio.solve.loop implements exactly this envelope; anything else
    # would silently take a code path that was dropped in the port
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
