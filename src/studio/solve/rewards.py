"""Reward wrappers on top of SPIDER's get_reward.

This is studio's contribution to the solve — the two places where
SPIDER's generic bimanual-manipulation reward is wrong for a G1 carrying
a reconstructed box. Ported from mppi_locoma's `retarget/rewards.py`.

1. Per-block weighted velocity tracking. DynaRetarget (arXiv 2602.06827)
   Table II tracks reference velocities next to positions; its ablation shows
   dropping the velocity terms costs 12.5 success points (74.6% -> 62.1%) and
   roughens trajectories (1.7 -> 2.7x ref jerk). SPIDER's built-in term is a
   single flat L2 over all nv dims, where the 35 robot dims numerically swamp
   the 6 object dims, so we replace it with a weighted one.
   config.vel_rew_scale acts as the master switch/scale: 0 keeps the term off,
   1.0 applies the relative weights below.

2. Simulated-box face-relative contact reward (make_box_face_contact_fns):
   palms target the nearest point on the grasp faces of the box where it
   ACTUALLY is, replacing SPIDER's contact_pos_ref world points baked from
   the reference box pose. make_reward_fns composes both.
"""

from __future__ import annotations

import torch
import warp as wp

from spider.config import Config
from spider.simulators.mjwp import (
    get_reward as _spider_get_reward,
    get_terminal_reward as _spider_get_terminal_reward,
)

# Relative weights per qvel block (humanoid_object nv layout: [0:3] base lin,
# [3:6] base ang, [6:-6] joints, [-6:-3] object lin, [-3:] object ang).
# Ratios follow DynaRetarget Table II rescaled to this repo's qpos weights
# (obj pos 8.0, joint 0.5). Object linear velocity dominates on purpose: its
# reference is exactly zero pre-grasp (penalizes knocking the box during
# approach) and it damps flinging/jerking during the lift.
VEL_WEIGHTS = {
    "base_lin": 0.06,
    "base_ang": 0.02,
    "joint": 0.02,
    "obj_lin": 0.15,
    "obj_ang": 0.015,
}


def make_weighted_vel_reward_fns(config: Config):
    """Return (get_reward, get_terminal_reward) with the weighted qvel term.

    Stashes and zeroes config.vel_rew_scale so SPIDER's flat full-nv term
    stays inert; the stashed value scales the weighted term instead. With
    vel_rew_scale == 0 the untouched SPIDER functions are returned.
    """
    vel_scale = float(config.vel_rew_scale)
    config.vel_rew_scale = 0.0
    if vel_scale == 0.0:
        return _spider_get_reward, _spider_get_terminal_reward

    if config.embodiment_type != "humanoid_object":
        raise ValueError(
            "weighted velocity reward only supports humanoid_object, got "
            f"{config.embodiment_type}"
        )
    w = torch.full((config.nv,), VEL_WEIGHTS["joint"], device=config.device)
    w[:3] = VEL_WEIGHTS["base_lin"]
    w[3:6] = VEL_WEIGHTS["base_ang"]
    w[-6:-3] = VEL_WEIGHTS["obj_lin"]
    w[-3:] = VEL_WEIGHTS["obj_ang"]
    w = w * vel_scale

    def get_reward(config, env, ref):
        rew, info = _spider_get_reward(config, env, ref)
        qvel_sim = wp.to_torch(env.data_wp.qvel)
        vel_rew = -torch.norm((qvel_sim - ref[1]) * w, p=2, dim=1)
        info["qvel_w_rew"] = vel_rew
        return rew + vel_rew, info

    def get_terminal_reward(config, env, ref):
        rew, info = get_reward(config, env, ref)
        return config.terminal_rew_scale * rew, info

    return get_reward, get_terminal_reward


def _quat_to_mat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Batched wxyz quaternion -> rotation matrix, (N, 4) -> (N, 3, 3)."""
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    m = torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        dim=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def make_box_face_contact_fns(
    config: Config,
    box_half,
    inner_fns,
    press: float = 0.0,
    edge_margin: float = 0.02,
):
    """Wrap (get_reward, get_terminal_reward) with a simulated-box face term.

    SPIDER's built-in contact reward pulls the palm sites to
    contact_pos_ref — world points baked from the REFERENCE box pose at
    scene-build time. Once the simulated box deviates, palms are dragged to
    where the box SHOULD be instead of where it is, fighting the physics.
    This term replaces it: each palm targets the nearest point on its grasp
    face of the box where it actually is (left palm -> +x_box, right ->
    -x_box: the emit_trial contact_pos/site-order convention), so the grip
    point on the face is implicitly optimized per sample and the reward
    keeps rewarding a firm hold on a drifted or rotated box.

    contact_rew_scale scales the term (SPIDER used it only as a gate) and
    the contact_ref mask still gates it to the grasp window. press pulls
    the target inside the face to bias squeezing (the scene's widened-box
    squeeze already does this geometrically); edge_margin keeps targets
    off the face edges so grips do not drift to corners.
    """
    scale = float(config.contact_rew_scale)
    if scale == 0.0 or len(config.contact_site_ids) != 2:
        return inner_fns  # SPIDER's built-in behavior stays as-is
    get_reward_in, _ = inner_fns
    config.contact_rew_scale = 0.0  # keep SPIDER's baked-point term inert

    hw, hd, hh = (float(v) for v in box_half)
    face_x = hw - press
    my = max(hd - edge_margin, 0.5 * hd)
    mz = max(hh - edge_margin, 0.5 * hh)
    sign = torch.tensor([1.0, -1.0], device=config.device)  # left, right

    def face_dist(env) -> torch.Tensor:
        qpos = wp.to_torch(env.data_wp.qpos)
        box_p = qpos[:, -7:-4]                                # (N, 3)
        rot = _quat_to_mat_wxyz(qpos[:, -4:])                 # box -> world
        site_xpos = wp.to_torch(env.data_wp.site_xpos)
        palms = site_xpos[:, config.contact_site_ids]          # (N, 2, 3)
        # world -> box frame: R^T (p - box_p)
        pl = torch.einsum("nji,nkj->nki", rot, palms - box_p.unsqueeze(1))
        tgt = torch.stack(
            [
                (sign * face_x).expand_as(pl[..., 0]),
                pl[..., 1].clamp(-my, my),
                pl[..., 2].clamp(-mz, mz),
            ],
            dim=-1,
        )
        return torch.norm(pl - tgt, dim=-1)                   # (N, 2)

    def get_reward(config, env, ref):
        rew, info = get_reward_in(config, env, ref)
        contact_ref = ref[3]                                  # (2,) 0/1 mask
        face_rew = -scale * (face_dist(env) * contact_ref.unsqueeze(0)).sum(
            dim=1
        )
        info["contact_face_rew"] = face_rew
        return rew + face_rew, info

    def get_terminal_reward(config, env, ref):
        rew, info = get_reward(config, env, ref)
        return config.terminal_rew_scale * rew, info

    return get_reward, get_terminal_reward


def make_reward_fns(config: Config, box_half=None):
    """Full reward stack: weighted velocity term, plus the simulated-box
    face contact term when box_half (the box geom's half extents) is given.
    Without box_half, SPIDER's baked contact_pos_ref term stays active."""
    fns = make_weighted_vel_reward_fns(config)
    if box_half is not None:
        fns = make_box_face_contact_fns(config, box_half, fns)
    return fns
