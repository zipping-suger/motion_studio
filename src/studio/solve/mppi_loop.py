"""The receding-horizon MPPI loop for obstacle-avoidance tasks.

Runs inside the solve venv, like its box-carry sibling `loop.py`:

    python -m studio.solve.mppi_loop --task-dir T [--param k=v ...]

where T is an under-table trial dir (scene.xml + task_info.json +
0/trajectory_kinematic.npz, see recon.table.emit_trial). Ported from the
mppi_obstacle experiment's mppi_solver.py (itself derived from
spider/examples/tutorial_collision_free_aug.ipynb).

Differences from `loop.py`: the embodiment is ``humanoid`` (the table is
static terrain, not a tracked object), the reward is replaced wholesale by
`obstacle_rewards`, there is no domain randomization, and diverged samples
are killed and resampled during rollout (terminate_resample). The result is
written as 0/trajectory_aug.npz; verification runs studio-side
(tasks.under_table), so this file stays solver-only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from spider.config import Config, compute_noise_schedule, compute_steps
from spider.interp import get_slice
from spider.io import load_data
from spider.optimizers.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
    make_rollout_fn,
)
from spider.simulators import mjwp

from ..tasks import kick_params, under_table_params
from ..tasks.under_table_params import coerce
from . import obstacle_rewards as rw

RESULT_NPZ = "trajectory_aug.npz"
CONFIG_JSON = "solve_config.json"

# per-task tunables: the trial's task_info.json names its task, which
# selects the solve defaults --param overrides are resolved against
TASK_PARAMS = {
    "under_table": under_table_params.SOLVE_DEFAULTS,
    "kick": kick_params.SOLVE_DEFAULTS,
}

# Fixed task setup (the tunables live in the per-task params modules).
SIM_DT = 1.0 / 60.0
CTRL_DT = 0.1
KNOT_DT = 0.1
IMPROVEMENT_THRESHOLD = 0.01


def resolve_params(cli_params: dict, task_type: str) -> dict:
    """The task's SOLVE_DEFAULTS + --param overrides, coerced to the
    defaults' types."""
    defaults = TASK_PARAMS.get(task_type)
    if defaults is None:
        raise SystemExit(f"no MPPI solve defaults for task {task_type!r} "
                         f"(known: {', '.join(TASK_PARAMS)})")
    out = dict(defaults)
    for key, val in cli_params.items():
        if key not in defaults:
            raise SystemExit(f"unknown solve parameter: {key}")
        out[key] = coerce(val, defaults[key])
    return out


def build_spider_config(params: dict, scene_xml: Path, traj_npz: Path,
                        ref_dt: float, n_frames: int, task: str) -> Config:
    """A spider Config for the injected G1 table scene — set directly, no
    hydra and no process_config (the dataset layout is ours, not spider's)."""
    mj_model = mujoco.MjModel.from_xml_path(str(scene_xml))

    config = Config(
        robot_type="unitree_g1",
        embodiment_type="humanoid",
        task=task,
        data_id=0,
        dataset_name="kimodo",
        viewer="",
        save_video=False,
        use_torch_compile=True,
    )

    # Timing
    config.sim_dt = SIM_DT
    config.ref_dt = ref_dt
    config.trace_dt = 2 * SIM_DT   # must be divisible by sim_dt
    config.render_dt = 4 * SIM_DT
    config.horizon = params["horizon"]
    config.ctrl_dt = CTRL_DT
    config.knot_dt = KNOT_DT

    # Sampling
    config.num_samples = params["num_samples"]
    config.max_num_iterations = params["max_num_iterations"]
    config.temperature = params["temperature"]
    config.improvement_threshold = IMPROVEMENT_THRESHOLD

    # Noise
    config.first_ctrl_noise_scale = params["first_ctrl_noise_scale"]
    config.last_ctrl_noise_scale = params["last_ctrl_noise_scale"]
    config.final_noise_scale = params["final_noise_scale"]
    config.joint_noise_scale = params["joint_noise_scale"]

    # Reward scales consumed by spider internals (tracking weights come
    # from obstacle_rewards, not from these)
    config.vel_rew_scale = params["vel_rew_scale"]
    config.contact_rew_scale = 0.0

    # Termination resampling — kill & replace diverged samples during rollout
    config.terminate_resample = params["terminate_resample"]
    config.base_pos_threshold = params["base_pos_threshold"]
    config.base_rot_threshold = params["base_rot_threshold"]

    config.sanity_check_seconds = 0.0

    # Paths & model dimensions
    config.model_path = str(scene_xml)
    config.data_path = str(traj_npz)
    config.nq = mj_model.nq
    config.nv = mj_model.nv
    config.nu = mj_model.nu
    config.npair = mj_model.npair
    config.nq_obj = 0

    compute_steps(config)
    compute_noise_schedule(config)

    # Damp exploration noise on the leg actuators: spider scales noise
    # uniformly across joints, but leg noise perturbs balance in every
    # rollout while the arms are the ones that need to explore around the
    # table. (noise_scale sample 0 stays zero — scaling preserves it.)
    if params["leg_noise_scale"] != 1.0:
        leg_kw = ("hip", "knee", "ankle")
        leg_ids = [
            i for i in range(mj_model.nu)
            if any(k in (mujoco.mj_id2name(
                mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "")
                for k in leg_kw)
        ]
        assert len(leg_ids) == 12, (
            f"expected 12 G1 leg actuators, found {len(leg_ids)}")
        config.noise_scale[:, :, leg_ids] *= params["leg_noise_scale"]

    # Solve through the whole motion: load_data pads the reference with
    # horizon + ctrl steps of repeated last frames for exactly this.
    config.max_sim_steps = int(n_frames * (config.ref_dt / config.sim_dt))
    return config


def solve(task_dir: Path, cli_params: dict) -> dict:
    """Run the full MPPI solve for one trial; writes 0/trajectory_aug.npz."""
    import torch  # solve venv only

    info = json.loads((task_dir / "task_info.json").read_text())
    params = resolve_params(cli_params, info.get("task_type", "under_table"))
    scene_xml = task_dir / "scene.xml"
    traj_npz = task_dir / "0" / "trajectory_kinematic.npz"
    n_frames = len(np.load(traj_npz)["qpos"])

    config = build_spider_config(params, scene_xml, traj_npz,
                                 info["ref_dt"], n_frames, task_dir.name)
    (task_dir / "0" / CONFIG_JSON).write_text(
        json.dumps(params, indent=2, default=str) + "\n")

    mj_model_scene = mujoco.MjModel.from_xml_path(str(scene_xml))
    robot_geoms = rw.RobotCollisionGeoms.resolve(
        mj_model_scene, info["robot_collision_geoms"])
    obstacle_ids = [
        gid for name in info["obstacle_geoms"]
        if (gid := mujoco.mj_name2id(mj_model_scene,
                                     mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0
    ]
    obstacles = rw.ObstacleSet(info["primitives"])
    get_reward, get_terminal_reward = rw.make_reward_fns(
        params, robot_geoms, obstacles, obstacle_ids)

    ref_data = load_data(config, config.data_path)
    qpos_ref_t, qvel_ref_t, ctrl_ref_t = ref_data[:3]

    env = mjwp.setup_env(config, ref_data)

    # CPU model mirrors the env state (time bookkeeping + sync_env)
    mj_model = mjwp.setup_mj_model(config)
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = qpos_ref_t[0].detach().cpu().numpy()
    mj_data.qvel[:] = qvel_ref_t[0].detach().cpu().numpy()
    mj_data.ctrl[:] = ctrl_ref_t[0].detach().cpu().numpy()
    mujoco.mj_step(mj_model, mj_data)
    mj_data.time = 0.0

    config.trace_site_ids = [
        sid for sid in range(mj_model.nsite)
        if (mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
            ).startswith("trace")
    ]
    # No domain randomization
    config.env_params_list = [
        [{"xy_offset": 0.0, "pair_margin": 0.0}]
        for _ in range(config.max_num_iterations)
    ]

    rollout = make_rollout_fn(
        mjwp.step_env, mjwp.save_state, mjwp.load_state,
        get_reward, get_terminal_reward, mjwp.get_terminate, mjwp.get_trace,
        mjwp.save_env_params, mjwp.load_env_params, mjwp.copy_sample_state,
    )
    optimize = make_optimize_fn(make_optimize_once_fn(rollout))

    ctrls = ctrl_ref_t[: config.horizon_steps]
    aug_qpos_list: list = []
    opt_steps_hist: list = []
    t_start = time.perf_counter()

    while True:
        t0 = time.perf_counter()
        sim_step = int(np.round(mj_data.time / config.sim_dt))
        ref_slice = get_slice(
            ref_data, sim_step + 1, sim_step + config.horizon_steps + 1)
        ctrls, infos = optimize(config, env, ctrls, ref_slice)

        # Execute the first ctrl_steps controls
        for i in range(config.ctrl_steps):
            mjwp.step_env(config, env, ctrls[i: i + 1])
            mj_data.qpos[:] = (
                mjwp.get_qpos(config, env)[0].detach().cpu().numpy())
            mj_data.qvel[:] = (
                mjwp.get_qvel(config, env)[0].detach().cpu().numpy())
            mj_data.ctrl[:] = ctrls[i].detach().cpu().numpy()
            mj_data.time += config.sim_dt
            aug_qpos_list.append(mj_data.qpos.copy())

        mjwp.sync_env(config, env, mj_data)

        # Receding horizon shift
        sim_step = int(np.round(mj_data.time / config.sim_dt))
        prev_ctrl = ctrls[config.ctrl_steps:]
        new_ctrl = ctrl_ref_t[
            sim_step + prev_ctrl.shape[0]:
            sim_step + prev_ctrl.shape[0] + config.ctrl_steps]
        ctrls = torch.cat([prev_ctrl, new_ctrl], dim=0)

        opt_steps_hist.append(int(np.asarray(infos["opt_steps"]).flat[0]))
        rtr = config.ctrl_dt / (time.perf_counter() - t0)
        print(f"Realtime rate: {rtr:.2f}, sim_steps: "
              f"{sim_step}/{config.max_sim_steps}, "
              f"opt_steps: {opt_steps_hist[-1]}", flush=True)

        if sim_step >= config.max_sim_steps:
            break

    elapsed = time.perf_counter() - t_start
    aug_qpos = (np.stack(aug_qpos_list, axis=0) if aug_qpos_list
                else np.zeros((0, config.nq)))
    n = len(aug_qpos)
    out_path = task_dir / "0" / RESULT_NPZ
    np.savez(
        out_path,
        aug_qpos=aug_qpos,                                   # (S, nq), 60 Hz
        ref_qpos_interp=qpos_ref_t[:n].detach().cpu().numpy(),
        sim_fps=1.0 / config.sim_dt,
        elapsed=elapsed,
        opt_steps=np.array(opt_steps_hist),
    )
    print(f"Final augmented trajectory: {n} frames in {elapsed:.1f}s "
          f"-> {out_path}", flush=True)
    return {"aug_qpos": aug_qpos, "elapsed": elapsed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task-dir", type=Path, required=True,
                    help="an under-table trial dir (see recon.table)")
    ap.add_argument("--param", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="solve hyperparameter override (repeatable)")
    args = ap.parse_args()

    cli_params = {}
    for item in args.param:
        key, _, value = item.partition("=")
        cli_params[key.strip()] = value.strip()
    solve(args.task_dir, cli_params)


if __name__ == "__main__":
    main()
