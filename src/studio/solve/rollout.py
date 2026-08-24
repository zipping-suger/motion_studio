"""SPIDER's rollout, ported to cut per-step overhead from the hot loop.

The rollout runs up to horizon_steps * max_num_iterations times per 0.1 s
control tick, and spider's version pays per sim step for three things
studio's headless solves never use: site-position traces, per-key info
stacking (reduced with a `.cpu()` sync per key per iteration), and
`terminate.any()` plus `.all()`. Semantics are unchanged — step, reward
accumulation, terminate-resample with its copy-over — but traces and info
are dropped and the resample decision costs one packed sync. The empty
info dict makes spider's optimize_once stats blocks no-ops, so the
upstream optimizer needs no changes.

Solve venv only (torch).
"""

from __future__ import annotations

import torch


def make_rollout_fn(step_env, save_state, load_state, get_reward,
                    get_terminal_reward, get_terminate,
                    save_env_params, load_env_params, copy_sample_state):
    """Same factory contract as spider's, minus the get_trace argument."""

    def rollout(config, env, ctrls, ref_slice, env_param):
        init_state = save_state(env)
        init_env_param = save_env_params(config, env)
        env = load_env_params(config, env, env_param)

        N, H = ctrls.shape[:2]
        cum_rew = torch.zeros(N, device=config.device)
        # unread when terminate_resample is off, so skip get_terminate
        terminate = torch.zeros(N, dtype=torch.bool, device=config.device)

        for t in range(H):
            step_env(config, env, ctrls[:, t])
            ref = [r[t] for r in ref_slice]
            rew, _info = (get_reward(config, env, ref) if t < H - 1
                          else get_terminal_reward(config, env, ref))
            cum_rew += rew

            if config.terminate_resample:
                terminate = get_terminate(config, env, ref)
                if t < H - 1:
                    # one packed sync instead of .any() plus .all()
                    n_bad = int(terminate.sum())
                    if 0 < n_bad < N:
                        bad = torch.nonzero(terminate).squeeze(-1)
                        good = torch.nonzero(~terminate).squeeze(-1)
                        if good.shape[0] > bad.shape[0]:
                            good = good[: bad.shape[0]]
                        elif good.shape[0] < bad.shape[0]:
                            random_idx = torch.randint(
                                0, good.shape[0], (bad.shape[0],))
                            good = good[random_idx]
                        # replace diverged samples: state, controls, reward
                        copy_sample_state(config, env, good, bad)
                        ctrls[bad, :t] = ctrls[good, :t]
                        cum_rew[bad] = cum_rew[good]

        mean_rew = cum_rew / H
        env = load_state(env, init_state)
        env = load_env_params(config, env, init_env_param)
        return ctrls, mean_rew, terminate, {}

    return rollout


def skip_zero_perturbation(config) -> None:
    """Patch mjwp's per-step perturbation round-trip out when it is a no-op.

    `apply_perturbation` round-trips the whole (N, nbody, 6) xfrc_applied
    array through torch every sim step; studio always solves unperturbed,
    where that changes nothing. step_env resolves the name from mjwp's
    globals at call time, so rebinding covers every call site."""
    if config.perturb_force or config.perturb_torque:
        return
    from spider.simulators import mjwp
    mjwp.apply_perturbation = lambda config, env: env
