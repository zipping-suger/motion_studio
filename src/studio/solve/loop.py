"""The receding-horizon SBMPC loop.

Runs inside the solve venv (`studio setup --solve`), which carries torch,
warp and the SPIDER wheel:

    python -m studio.solve.loop --task T --dataset-dir D [--param k=v ...]

Derived from SPIDER's `examples/run_mjwp.py`, trimmed to the one
configuration studio uses — `humanoid_object`, headless, no contact
guidance, no Gibbs sampling, no video. `spider_cfg.build` asserts that
envelope.

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
import re
import time
from pathlib import Path

import mujoco
import numpy as np

from spider.interp import get_slice
from spider.io import load_data
from spider.optimizers.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
)
from spider.postprocess.get_success_rate import compute_object_tracking_error
from spider.simulators.mjwp import (
    copy_sample_state,
    get_qpos,
    get_qvel,
    get_terminate,
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
from .rollout import make_rollout_fn, skip_zero_perturbation
from .spec import SolveScene

RESULT_NPZ = "trajectory_mjwp.npz"
CONFIG_JSON = "solve_config.json"

# endpoint-shaped object tracking (rewards: ref[5]): full end weight for
# this long past the grab / before the release, then a linear blend down
END_HOLD_S = 0.4
END_RAMP_S = 0.4


def _obj_weight_profile(config, task_info, n_steps):
    """Per-sim-step multiplier on the object tracking error.

    The pick and release frames weigh obj_end_rew_scale, the transit
    between them obj_mid_rew_scale. Flat at the end scale without
    pick/release frames in task_info, or with equal scales."""
    import torch
    end = float(getattr(config, "obj_end_rew_scale", 1.0))
    mid = float(getattr(config, "obj_mid_rew_scale", 1.0))
    grab, release = task_info.get("pick_frame"), task_info.get("release_frame")
    if grab is None or release is None or end == mid:
        return torch.full((n_steps,), end, device=config.device)
    k = float(task_info.get("ref_dt", config.sim_dt)) / config.sim_dt
    hold = END_HOLD_S / config.sim_dt
    ramp = max(END_RAMP_S / config.sim_dt, 1.0)
    t = torch.arange(n_steps, device=config.device, dtype=torch.float32)
    # how far into the mid region each step sits, in ramp units: <= 0 at
    # and outside the endpoints, saturating to 1 deep in the transit
    into_mid = torch.minimum(t - (grab * k + hold),
                             (release * k - hold) - t) / ramp
    return end + (mid - end) * into_mid.clamp(0.0, 1.0)


def _support_channel(config, task_info, mj_model, n_steps):
    """(geom -> support group lookup, group count, per-step gates (n,
    G)) from task_info's `supports`: one group per body geom the
    reference rests the object on, its gate 1 over the frames it does."""
    import torch
    sups = task_info.get("supports") or []
    groups = sorted({s["geom"] for s in sups})
    geom_supp = np.full(mj_model.ngeom, -1, dtype=np.int64)
    for gi, name in enumerate(groups):
        g = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g >= 0:
            geom_supp[g] = gi
    gates = torch.zeros((n_steps, max(len(groups), 1)), device=config.device)
    k = float(task_info.get("ref_dt", config.sim_dt)) / config.sim_dt
    for s_ in sups:
        gi = groups.index(s_["geom"])
        a = int(round(s_["start"] * k))
        b = min(int(round((s_["end"] + 1) * k)), n_steps)
        gates[a:b, gi] = 1.0
    return geom_supp, len(groups), gates


def _env_params(config) -> list[list[dict]]:
    """Per-iteration domain-randomization parameter sets.

    num_dr == 1 means the nominal midpoint, not the ranges' low endpoint
    that np.linspace(n=1) would return.
    """
    assert config.num_dr >= 1, "num_dr must be >= 1 or nothing gets rolled out"

    def spread(lo_hi):
        if config.num_dr == 1:
            return np.array([float(np.mean(lo_hi))])
        return np.linspace(*lo_hi, config.num_dr)

    xy = spread(config.xy_offset_range)
    margin = spread(config.pair_margin_range)
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

    # a CPU MuJoCo model alongside the warp worlds: it owns the receding
    # horizon's clock and is where qpos is read back to
    mj_model = setup_mj_model(config)
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mj_data.qvel[:] = qvel_ref[0].detach().cpu().numpy()
    mj_data.ctrl[:] = ctrl_ref[0].detach().cpu().numpy()
    mujoco.mj_step(mj_model, mj_data)
    mj_data.time = 0.0

    config.env_params_list = _env_params(config)

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(config.output_dir) / CONFIG_JSON).write_text(
        json.dumps(spider_cfg.summarize(config), indent=2, default=str) + "\n")

    # everything task-specific about this solve arrives as data in the
    # trial's task_info (see spec.SolveScene); no task is known by name
    info_path = Path(config.model_path).parent / "task_info.json"
    task_info = (json.loads(info_path.read_text())
                 if info_path.exists() else {})
    scene = SolveScene.from_info(task_info)

    # box_half turns on the face contact term: palms target the SIMULATED
    # object's grasp faces, which only a trial declaring `object.faces`
    # has (a one-hand pick reuses the largebox names for an object where
    # both palms on side faces would be wrong)
    box_gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM,
                                "largebox_geom")
    box_half = (mj_model.geom_size[box_gid].copy()
                if box_gid >= 0 and scene.faces else None)

    # damp finger-actuator exploration when the trial asks: sampled
    # wiggling pries a held object out of the grasp
    hand_scale = scene.hand_noise_scale
    if hand_scale is not None:
        hand_kw = ("thumb", "index", "middle", "ring", "pinky")
        hand_ids = [
            a for a in range(mj_model.nu)
            if any(k in (mujoco.mj_id2name(
                mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "")
                for k in hand_kw)]
        if hand_ids:
            config.noise_scale[:, :, hand_ids] *= float(hand_scale)

    # grip reward targets: each holding hand's calibrated pocket and the
    # object-frame anchor it holds (None = the object's origin — a small
    # picked object; a pole anchors its grab height up the axis, a chair
    # each hand's grab point in the chair's frame)
    grips = scene.grips or None

    # the template's ghost self-collision pairs (forceless, solver-free)
    # feed the self-penetration penalty
    self_pairs = [
        (int(mj_model.pair_geom1[p]), int(mj_model.pair_geom2[p]))
        for p in range(mj_model.npair)
        if (mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_PAIR, p) or ""
            ).startswith("self_")]

    # the grasp term's geom lookups: palm boxes + finger capsules per
    # hand column, and every object collision geom (the box/handle
    # capsule, or a chair's hulls)
    geom_hand = np.full(mj_model.ngeom, -1, dtype=np.int64)
    geom_obj = np.zeros(mj_model.ngeom, dtype=bool)
    for g in range(mj_model.ngeom):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if re.fullmatch(r"[lr]h(_\w+)?", name):
            geom_hand[g] = 0 if name[0] == "l" else 1
        elif name == "largebox_geom" or name.startswith("largebox_c"):
            geom_obj[g] = True

    # body supports (a chair on the hip, a box on the forearms): the
    # geoms the reference rests the object on, gated over their windows
    geom_supp, n_supp, supp_gates = _support_channel(
        config, task_info, mj_model, qpos_ref.shape[0])

    reward_fn, terminal_reward_fn = make_reward_fns(
        config, box_half=box_half, grips=grips, self_pairs=self_pairs,
        grasp=(geom_hand, geom_obj), support=(geom_supp, n_supp))

    # the per-step object weight rides along as a 6th reference channel,
    # the support gates as a 7th; setup_env above already took the
    # 5-tuple SPIDER expects
    ref_data = (*ref_data,
                _obj_weight_profile(config, task_info, qpos_ref.shape[0]),
                supp_gates)

    skip_zero_perturbation(config)
    rollout = make_rollout_fn(
        step_env, save_state, load_state, reward_fn, terminal_reward_fn,
        # spider's terminate check unpacks exactly five ref tensors
        lambda c, e, r: get_terminate(c, e, r[:5]),
        save_env_params, load_env_params,
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
