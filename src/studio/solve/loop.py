"""The receding-horizon SBMPC loop.

Runs inside the solve venv (`studio setup --solve`), which carries torch,
warp and the SPIDER wheel:

    python -m studio.solve.loop --task T --dataset-dir D [--param k=v ...]

Ported from SPIDER's `examples/run_mjwp.py` by way of mppi_locoma, trimmed
to the one configuration studio uses — `humanoid_object`, headless, no
contact guidance, no Gibbs sampling, no video. `spider_cfg.build` asserts
that envelope, so a config outside it fails loudly instead of silently
taking a path this file dropped.

Three nested loops:

  outer   receding horizon — optimize over a 1.0 s reference window, then
          commit only ctrl_steps (6) of it and shift the buffer forward
  middle  annealing — max_num_iterations passes per tick, exploration noise
          decaying as beta_traj ** i (SPIDER's make_optimize_fn)
  inner   one MPPI update — perturb spline knots, roll out num_samples
          worlds in parallel, softmax-weight the top 10%, average

Reward comes from studio.solve.rewards, layered over SPIDER's.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from spider.interp import get_slice
from spider.io import load_data
from spider.optimizers.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
    make_rollout_fn,
)
from spider.postprocess.get_success_rate import compute_object_tracking_error
from spider.simulators.mjwp import (
    copy_sample_state,
    get_qpos,
    get_qvel,
    get_terminate,
    get_trace,
    load_env_params,
    load_state,
    save_env_params,
    save_state,
    setup_env,
    setup_mj_model,
    step_env,
    sync_env,
)

from . import spider_cfg
from .rewards import make_reward_fns

RESULT_NPZ = "trajectory_mjwp.npz"
CONFIG_JSON = "solve_config.json"


def _trace_site_ids(mj_model) -> list[int]:
    """Sites named trace* — object ones first, matching SPIDER's ordering."""
    object_ids, robot_ids = [], []
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name and name.startswith("trace"):
            (object_ids if "object" in name else robot_ids).append(sid)
    return object_ids + robot_ids


def _env_params(config) -> list[list[dict]]:
    """Per-iteration domain-randomization parameter sets.

    num_dr == 1 means the margin/offset endpoints are applied but not
    varied, which is what mppi_locoma ran.
    """
    assert config.num_dr >= 1, "num_dr must be >= 1 or nothing gets rolled out"
    xy = np.linspace(*config.xy_offset_range, config.num_dr)
    margin = np.linspace(*config.pair_margin_range, config.num_dr)
    per_iter = [{"xy_offset": xy[j], "pair_margin": margin[j]}
                for j in range(config.num_dr)]
    return [list(per_iter) for _ in range(config.max_num_iterations)]


def solve(config) -> dict:
    """Run the full solve for one trial. Writes trajectory_mjwp.npz into
    config.output_dir and returns the object tracking errors."""
    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(
        config, config.data_path)
    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)
    if config.max_sim_steps <= 0:
        config.max_sim_steps = (
            qpos_ref.shape[0] - config.horizon_steps - config.ctrl_steps)

    env = setup_env(config, ref_data)

    # a CPU MuJoCo model alongside the warp worlds: it owns the clock that
    # drives the receding horizon, and it is where qpos is read back to
    mj_model = setup_mj_model(config)
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mj_data.qvel[:] = qvel_ref[0].detach().cpu().numpy()
    mj_data.ctrl[:] = ctrl_ref[0].detach().cpu().numpy()
    mujoco.mj_step(mj_model, mj_data)
    mj_data.time = 0.0

    config.trace_site_ids = _trace_site_ids(mj_model)
    config.env_params_list = _env_params(config)

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(config.output_dir) / CONFIG_JSON).write_text(
        json.dumps(spider_cfg.summarize(config), indent=2, default=str) + "\n")

    # palms target the SIMULATED box's grasp faces (half extents read off
    # the model), not the baked reference contact points
    box_gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM,
                                "largebox_geom")
    box_half = mj_model.geom_size[box_gid].copy() if box_gid >= 0 else None
    reward_fn, terminal_reward_fn = make_reward_fns(config, box_half=box_half)

    rollout = make_rollout_fn(
        step_env, save_state, load_state, reward_fn, terminal_reward_fn,
        get_terminate, get_trace, save_env_params, load_env_params,
        copy_sample_state,
    )
    optimize = make_optimize_fn(make_optimize_once_fn(rollout))

    ctrls = ctrl_ref[: config.horizon_steps]
    info_list: list[dict] = []
    t_start = time.perf_counter()

    while True:
        t0 = time.perf_counter()

        # optimize against the future reference window (+1 lookahead)
        sim_step = int(np.round(mj_data.time / config.sim_dt))
        ref_slice = get_slice(
            ref_data, sim_step + 1, sim_step + config.horizon_steps + 1)
        ctrls, infos = optimize(config, env, ctrls, ref_slice)

        # commit only the first ctrl_steps of the plan
        step_info = {"qpos": [], "qvel": [], "time": [], "ctrl": []}
        for i in range(config.ctrl_steps):
            ctrl_step = ctrls[i]
            step_env(config, env, ctrl_step)
            mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
            mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
            mj_data.ctrl[:] = ctrl_step.detach().cpu().numpy()
            mj_data.time += config.sim_dt
            step_info["qpos"].append(mj_data.qpos.copy())
            step_info["qvel"].append(mj_data.qvel.copy())
            step_info["time"].append(mj_data.time)
            step_info["ctrl"].append(mj_data.ctrl.copy())
        infos.update({k: np.stack(v, axis=0) for k, v in step_info.items()})
        sync_env(config, env, mj_data)

        # receding horizon: keep the untouched tail, refill from the reference
        sim_step = int(np.round(mj_data.time / config.sim_dt))
        prev_ctrl = ctrls[config.ctrl_steps:]
        new_ctrl = ctrl_ref[
            sim_step + prev_ctrl.shape[0]:
            sim_step + prev_ctrl.shape[0] + config.ctrl_steps]
        import torch  # local: keeps module import cheap for --help
        ctrls = torch.cat([prev_ctrl, new_ctrl], dim=0)

        rtr = config.ctrl_dt / (time.perf_counter() - t0)
        print(f"Realtime rate: {rtr:.2f}, plan time: "
              f"{time.perf_counter() - t0:.4f}s, sim_steps: {sim_step}/"
              f"{config.max_sim_steps}, opt_steps: {infos['opt_steps'][0]}",
              flush=True)

        info_list.append({k: v for k, v in infos.items() if k != "trace_sample"})
        if sim_step >= config.max_sim_steps:
            break

    print(f"Total time: {time.perf_counter() - t_start:.4f}s", flush=True)

    if not info_list:
        raise SystemExit("solve produced no control ticks")

    aggregated = {k: np.stack([info[k] for info in info_list], axis=0)
                  for k in info_list[0]}
    out_path = Path(config.output_dir) / RESULT_NPZ
    np.savez(out_path, **aggregated)

    qpos_traj = np.concatenate([info["qpos"] for info in info_list], axis=0)
    errors = compute_object_tracking_error(
        qpos_traj,
        qpos_ref[: qpos_traj.shape[0]].detach().cpu().numpy(),
        config.embodiment_type, "mjwp")
    print(f"Final object tracking error: pos={errors['obj_pos_err']:.4f}, "
          f"quat={errors['obj_quat_err']:.4f}", flush=True)
    print(f"Saved {out_path}", flush=True)
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, help="trial/task name")
    ap.add_argument("--dataset-dir", type=Path, required=True,
                    help="a run's outputs/ dir (holds processed/...)")
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                    help="solve hyperparameter override (repeatable)")
    args = ap.parse_args()

    params = {}
    for item in args.param:
        key, _, value = item.partition("=")
        params[key.strip()] = value.strip()
    solve(spider_cfg.build(args.task, args.dataset_dir, params))


if __name__ == "__main__":
    main()
