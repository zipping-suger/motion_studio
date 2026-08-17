"""Obstacle-avoidance MPPI reward: tracking + stability + SDF + collisions.

The under-table counterpart of `rewards.py`: instead of layering object
terms over SPIDER's humanoid_object reward, this replaces the reward
outright for the robot-only (``humanoid``) embodiment — weighted qpos
tracking, dedicated base-drift penalties, a reference-relative stability
term, and analytic-SDF obstacle avoidance with a per-sample collision
count.

Ported from the mppi_obstacle experiment's rewards.py (itself derived from
spider/examples/tutorial_collision_free_aug.ipynb). Runs in the solve venv
(torch + warp + SPIDER); reward weights arrive as the flat params dict of
`tasks.under_table_params.SOLVE_DEFAULTS`.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch
import warp as wp

# Reuse spider's quaternion-correct qpos difference (humanoid branch).
from spider.simulators.mjwp import _diff_qpos


def build_qpos_weight(nv: int, p: dict, device) -> torch.Tensor:
    """Per-DOF tracking weights for the G1 humanoid (nv=35).

    Layout: [0:3] base pos, [3:6] base rot, [6:18] legs, [18:21] waist,
    [21:25]/[28:32] shoulder+elbow, [25:28]/[32:35] wrist.
    """
    assert nv == 35, f"G1 humanoid expects nv=35, got {nv}"
    w = torch.ones(nv, device=device)
    w[0:3] = p["qw_base_pos"]
    w[3:6] = p["qw_base_rot"]
    w[6:18] = p["qw_legs"]
    w[18:21] = p["qw_waist"]
    w[21:25] = p["qw_shoulder_elbow"]  # left shoulder + elbow
    w[25:28] = p["qw_wrist"]           # left wrist — hand pose
    w[28:32] = p["qw_shoulder_elbow"]  # right shoulder + elbow
    w[32:35] = p["qw_wrist"]           # right wrist
    return w


def _point_box_sdf(points, center, half_extents, yaw):
    d = points - center
    c, s = torch.cos(yaw), torch.sin(yaw)
    dx = d[..., 0] * c + d[..., 1] * s
    dy = -d[..., 0] * s + d[..., 1] * c
    local = torch.stack([dx, dy, d[..., 2]], dim=-1)
    q = torch.abs(local) - half_extents
    outside = torch.norm(torch.clamp(q, min=0.0), dim=-1)
    inside = torch.clamp(q.max(dim=-1).values, max=0.0)
    return outside + inside


class ObstacleSet:
    """Analytic SDF over scene box primitives, with per-device caching."""

    def __init__(self, primitives: list):
        # primitives: [(type, pos[3], [hx, hy, hz, yaw]), ...] from
        # task_info.json (recon.table.TableSpec.primitives)
        self.primitives = primitives
        self._cache: dict = {}

    def _tensors(self, device):
        if device not in self._cache:
            self._cache[device] = [
                (torch.tensor(pos, device=device, dtype=torch.float32),
                 torch.tensor(params[:3], device=device, dtype=torch.float32),
                 torch.tensor(params[3], device=device, dtype=torch.float32))
                for _obs_type, pos, params in self.primitives
            ]
        return self._cache[device]

    def min_sdf(self, points: torch.Tensor) -> torch.Tensor:
        """Minimum signed distance from each point to any primitive."""
        min_sdf = torch.full(points.shape[:-1], 1e6, device=points.device)
        for center, half, yaw in self._tensors(points.device):
            min_sdf = torch.minimum(min_sdf,
                                    _point_box_sdf(points, center, half, yaw))
        return min_sdf


@dataclass
class RobotCollisionGeoms:
    """Robot collision geoms resolved from the injected scene model."""

    ids: list
    radii: list
    half_lengths: list
    is_capsule: list

    @classmethod
    def resolve(cls, mj_model: mujoco.MjModel, geom_names: list):
        ids, radii, half_lengths, is_capsule = [], [], [], []
        for name in geom_names:
            gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                continue
            ids.append(gid)
            radii.append(float(mj_model.geom_size[gid, 0]))
            cap = mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CAPSULE
            is_capsule.append(bool(cap))
            half_lengths.append(float(mj_model.geom_size[gid, 1]) if cap
                                else 0.0)
        return cls(ids, radii, half_lengths, is_capsule)


def _stability_reward(qpos_sim: torch.Tensor,
                      qpos_ref: torch.Tensor) -> torch.Tensor:
    """Penalize pelvis height drop and torso tilt beyond the reference."""
    N = qpos_sim.shape[0]
    pelvis_z_ref = qpos_ref[2].unsqueeze(0).expand(N)
    height_drop = torch.clamp(pelvis_z_ref - qpos_sim[:, 2], min=0.0)
    height_penalty = height_drop**2

    qx, qy = qpos_sim[:, 4], qpos_sim[:, 5]
    up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    # Reference-relative, not absolute: the under-table crouch tilts the
    # pelvis up to ~39 degrees in the reference itself — an absolute term
    # costs a constant ~11 (at weight 50) for tracking correctly and pushes
    # the solve to straighten out of the crouch. Only tilting MORE than the
    # reference at the same frame is penalized. Linear, not squared: a
    # 15-degree extra lean gives up_ref-up_z ~ 0.034; squared that is
    # ~0.001, invisible next to tracking until the robot has tipped.
    up_ref = 1.0 - 2.0 * (qpos_ref[4] ** 2 + qpos_ref[5] ** 2)
    tilt_penalty = torch.clamp(up_ref - up_z, min=0.0)

    return -(height_penalty + tilt_penalty)


def make_reward_fns(params: dict, robot_geoms: RobotCollisionGeoms,
                    obstacles: ObstacleSet, obstacle_geom_ids: list):
    """Build (get_reward, get_terminal_reward) for spider's make_rollout_fn.

    ``params`` is the resolved under_table solve dict; robot_geoms /
    obstacles / obstacle_geom_ids describe the injected scene.
    """
    weight_cache: dict = {}
    geom_cache: dict = {}
    obs_ids_cache: dict = {}
    obs_ids_cpu = torch.tensor(obstacle_geom_ids, dtype=torch.int32)

    def _geom_tensors(device):
        """Index tensors for batched point assembly: capsules first, then
        the rest. Point layout is [cap centers | cap +ends | cap -ends |
        other centers], with radii ordered to match."""
        if device not in geom_cache:
            cap_ids, cap_r, cap_hl, other_ids, other_r = [], [], [], [], []
            for i, gid in enumerate(robot_geoms.ids):
                if robot_geoms.is_capsule[i]:
                    cap_ids.append(gid)
                    cap_r.append(robot_geoms.radii[i])
                    cap_hl.append(robot_geoms.half_lengths[i])
                else:
                    other_ids.append(gid)
                    other_r.append(robot_geoms.radii[i])
            cap_r_t = torch.tensor(cap_r, device=device, dtype=torch.float32)
            geom_cache[device] = (
                torch.tensor(cap_ids, device=device, dtype=torch.long),
                torch.tensor(cap_hl, device=device,
                             dtype=torch.float32).view(1, -1, 1),
                torch.tensor(other_ids, device=device, dtype=torch.long),
                torch.cat([
                    cap_r_t, cap_r_t, cap_r_t,
                    torch.tensor(other_r, device=device, dtype=torch.float32),
                ]),
            )
        return geom_cache[device]

    def _sdf_collision_reward(env, device):
        """Soft SDF avoidance: hinge penalty within collision_margin.

        Robot collision capsules are sampled at center + both axis
        endpoints; each sample point must stay margin + radius away from
        every obstacle.
        """
        geom_xpos = wp.to_torch(env.data_wp.geom_xpos)
        geom_xmat = wp.to_torch(env.data_wp.geom_xmat)
        N = geom_xpos.shape[0]
        cap_ids, cap_hl, other_ids, radii = _geom_tensors(device)

        cap_centers = geom_xpos[:, cap_ids, :]  # (N, C, 3)
        # capsule z-axis = third column of the rotation matrix
        cap_axes = geom_xmat[:, cap_ids, :].reshape(N, -1, 3, 3)[..., 2]
        ends = cap_hl * cap_axes
        points = torch.cat(
            [cap_centers, cap_centers + ends, cap_centers - ends,
             geom_xpos[:, other_ids, :]],
            dim=1,
        )

        clearance = obstacles.min_sdf(points) - radii[None, :]
        violation = torch.clamp(params["collision_margin"] - clearance,
                                min=0.0)
        # Two-tier hinge: the outer linear hinge equilibrates against
        # tracking cost right at the surface (grazing); the steeper inner
        # hinge makes the last collision_inner_margin metres
        # disproportionately expensive so solutions keep real clearance.
        if params["collision_inner_margin"] > 0.0:
            inner = torch.clamp(params["collision_inner_margin"] - clearance,
                                min=0.0)
            violation = violation + params["collision_inner_scale"] * inner
        return -violation.sum(dim=1)

    def _collision_count_penalty(env, num_samples, device):
        """-1 per penetrating robot-obstacle contact per sample.

        Works with ghost obstacles too: the pairs' large ``gap`` removes
        contacts from the solver but they are still written to the contact
        arrays.
        """
        contact_geom = wp.to_torch(env.data_wp.contact.geom)        # (ncon, 2)
        contact_dist = wp.to_torch(env.data_wp.contact.dist)
        contact_dim = wp.to_torch(env.data_wp.contact.dim)
        contact_worldid = wp.to_torch(env.data_wp.contact.worldid)

        if device not in obs_ids_cache:
            obs_ids_cache[device] = obs_ids_cpu.to(device)
        obs_ids = obs_ids_cache[device]

        active = contact_dim > 0
        is_obs = torch.isin(contact_geom[:, 0], obs_ids) | torch.isin(
            contact_geom[:, 1], obs_ids)
        valid = active & is_obs & (contact_dist < 0.0)

        counts = torch.zeros(num_samples, device=device)
        if valid.any():
            wids = contact_worldid[valid].long()
            wids = wids[(wids >= 0) & (wids < num_samples)]
            if wids.numel() > 0:
                counts.scatter_add_(0, wids,
                                    torch.ones(wids.shape[0], device=device))
        return -counts

    def get_reward(config, env, ref):
        qpos_ref, qvel_ref = ref[0], ref[1]
        qpos_sim = wp.to_torch(env.data_wp.qpos)
        qvel_sim = wp.to_torch(env.data_wp.qvel)
        device = config.device

        # Weighted qpos tracking (mirrors mjwp.get_reward, custom weights)
        qpos_diff = _diff_qpos(
            config, qpos_sim,
            qpos_ref.unsqueeze(0).repeat(qpos_sim.shape[0], 1))
        if device not in weight_cache:
            weight_cache[device] = build_qpos_weight(config.nv, params, device)
        qpos_dist = torch.norm(qpos_diff * weight_cache[device], p=2, dim=1)
        qvel_dist = torch.norm(qvel_sim - qvel_ref, p=2, dim=1)
        qpos_rew = -qpos_dist
        qvel_rew = -config.vel_rew_scale * qvel_dist

        # Dedicated base-drift penalties, quadratic and OUTSIDE the coupled
        # tracking norm: the norm's sqrt compresses marginal base error
        # whenever the arm deviates for avoidance, making base drift the
        # cheapest error to trade away. Quadratic growth makes accumulating
        # drift increasingly expensive instead.
        base_pos_pen = (qpos_diff[:, :3] ** 2).sum(dim=1)
        base_rot_pen = (qpos_diff[:, 3:6] ** 2).sum(dim=1)
        # Horizontal base velocity error: catches a forward trip while it is
        # still momentum, before it becomes displacement.
        base_vel_pen = torch.norm(qvel_sim[:, :2] - qvel_ref[:2], p=2, dim=1)
        # Arm joint-velocity error (DOF 21:35): damps the press-shove-press
        # oscillation where wrist tracking and contact/SDF forces fight at
        # the table.
        arm_vel_pen = torch.norm(qvel_sim[:, 21:35] - qvel_ref[21:35],
                                 p=2, dim=1)

        stab_rew = _stability_reward(qpos_sim, qpos_ref)
        if params["sdf_weight"] > 0:
            sdf_rew = _sdf_collision_reward(env, device)
        else:
            sdf_rew = torch.zeros_like(qpos_rew)
        collision_pen = _collision_count_penalty(env, config.num_samples,
                                                 device)

        rew = (
            qpos_rew
            + qvel_rew
            - params["base_pos_drift_weight"] * base_pos_pen
            - params["base_rot_drift_weight"] * base_rot_pen
            - params["base_vel_weight"] * base_vel_pen
            - params["arm_vel_weight"] * arm_vel_pen
            + params["stability_weight"] * stab_rew
            + params["sdf_weight"] * sdf_rew
            + params["collision_count_weight"] * collision_pen
        )

        info = {
            "qpos_dist": qpos_dist,
            "qvel_dist": qvel_dist,
            "qpos_rew": qpos_rew,
            "qvel_rew": qvel_rew,
            "base_pos_pen": base_pos_pen,
            "base_rot_pen": base_rot_pen,
            "base_vel_pen": base_vel_pen,
            "arm_vel_pen": arm_vel_pen,
            "stab_rew": stab_rew,
            "sdf_rew": sdf_rew,
            "collision_count": -collision_pen,  # positive = collisions
        }
        return rew, info

    def get_terminal_reward(config, env, ref):
        rew, info = get_reward(config, env, ref)
        return config.terminal_rew_scale * rew, info

    return get_reward, get_terminal_reward
