"""Obstacle-avoidance MPPI reward: tracking + stability + SDF + collisions.

The under-table counterpart of `rewards.py`. Rather than layering object
terms over SPIDER's humanoid_object reward, this replaces the reward
outright for the robot-only (``humanoid``) embodiment. Weights arrive as
the flat params dict of `tasks.under_table_params.SOLVE_DEFAULTS`.

Solve venv only (torch + warp + SPIDER).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch
import warp as wp

from .quat import quat_sub


# joint-name keywords -> tracking-weight param; first match wins
_QW_GROUPS = (
    (("hip", "knee", "ankle"), "qw_legs"),
    (("waist",), "qw_waist"),
    (("shoulder", "elbow"), "qw_shoulder_elbow"),
    (("wrist",), "qw_wrist"),
    (("thumb", "index", "middle", "ring", "pinky"), "qw_hands"),
)

# joints the arm damping term covers (fingers included: zero reference
# velocity, so damping them is free)
_ARM_VEL_KEYWORDS = ("shoulder", "elbow", "wrist",
                     "thumb", "index", "middle", "ring", "pinky")


def _hinge_joints(mj_model):
    """(name, dofadr) for every hinge joint, whatever the layout."""
    out = []
    for j in range(mj_model.njnt):
        if mj_model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        out.append((name, int(mj_model.jnt_dofadr[j])))
    return out


def build_qpos_weight(mj_model, p: dict, device) -> torch.Tensor:
    """Per-DOF tracking weights, classified by joint name so the layout
    (with or without the BrainCo hand joints) never matters."""
    w = torch.ones(mj_model.nv, device=device)
    w[0:3] = p["qw_base_pos"]
    w[3:6] = p["qw_base_rot"]
    for name, adr in _hinge_joints(mj_model):
        for keywords, param in _QW_GROUPS:
            if any(k in name for k in keywords):
                w[adr] = p[param]
                break
    return w


def arm_dof_indices(mj_model) -> list:
    """dof addresses of the arm and finger joints."""
    return [adr for name, adr in _hinge_joints(mj_model)
            if any(k in name for k in _ARM_VEL_KEYWORDS)]


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
        # stacked over the K primitives: min_sdf is then one batched op
        if device not in self._cache:
            self._cache[device] = (
                torch.tensor([pos for _t, pos, _p in self.primitives],
                             device=device, dtype=torch.float32),
                torch.tensor([params[:3] for _t, _pos, params
                              in self.primitives],
                             device=device, dtype=torch.float32),
                torch.tensor([params[3] for _t, _pos, params
                              in self.primitives],
                             device=device, dtype=torch.float32),
            )
        return self._cache[device]

    def min_sdf(self, points: torch.Tensor) -> torch.Tensor:
        """Minimum signed distance from each point to any primitive."""
        if not self.primitives:
            return torch.full(points.shape[:-1], 1e6, device=points.device)
        centers, halves, yaws = self._tensors(points.device)
        # (..., 1, 3) against (K, 3)/(K,) broadcasts to a (..., K) SDF
        sdf = _point_box_sdf(points.unsqueeze(-2), centers, halves, yaws)
        return sdf.min(dim=-1).values


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
            gtype = mj_model.geom_type[gid]
            if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                is_capsule.append(True)
                radii.append(float(mj_model.geom_size[gid, 0]))
                half_lengths.append(float(mj_model.geom_size[gid, 1]))
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                # e.g. the BrainCo palm: sample as a capsule along the
                # box's z (long) axis, radius from the lateral extents
                is_capsule.append(True)
                radii.append(float(mj_model.geom_size[gid, :2].max()))
                half_lengths.append(float(mj_model.geom_size[gid, 2]))
            else:
                is_capsule.append(False)
                radii.append(float(mj_model.geom_size[gid, 0]))
                half_lengths.append(0.0)
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
    # Reference-relative: the under-table crouch tilts the pelvis ~39
    # degrees in the reference itself, so an absolute term would push the
    # solve to straighten out of it. Linear, not squared — squared, a 15
    # degree extra lean is invisible next to tracking until it has tipped.
    up_ref = 1.0 - 2.0 * (qpos_ref[4] ** 2 + qpos_ref[5] ** 2)
    tilt_penalty = torch.clamp(up_ref - up_z, min=0.0)

    return -(height_penalty + tilt_penalty)


def get_terminate(config, env, ref):
    """mjwp.get_terminate's humanoid branch with the sync-free quat_sub;
    the terminate-resample path evaluates it every rollout sim step."""
    qpos_sim = wp.to_torch(env.data_wp.qpos)
    qpos_ref = ref[0]
    pos_err = torch.norm(qpos_sim[:, :3] - qpos_ref[:3], p=2, dim=1)
    rot_err = torch.norm(quat_sub(qpos_sim[:, 3:7], qpos_ref[3:7]),
                         p=2, dim=1)
    return ((pos_err > config.base_pos_threshold)
            | (rot_err > config.base_rot_threshold))


def make_reward_fns(params: dict, mj_model, robot_geoms: RobotCollisionGeoms,
                    obstacles: ObstacleSet, obstacle_geom_ids: list):
    """Build (get_reward, get_terminal_reward) for spider's make_rollout_fn.

    ``params`` is the resolved under_table solve dict; the rest describe
    the injected scene model.
    """
    weight_cache: dict = {}
    geom_cache: dict = {}
    obs_ids_cache: dict = {}
    arm_idx_cache: dict = {}
    obs_ids_cpu = torch.tensor(obstacle_geom_ids, dtype=torch.int32)
    arm_idx_cpu = torch.tensor(arm_dof_indices(mj_model), dtype=torch.long)

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
        # two-tier hinge: the outer one equilibrates against tracking cost
        # at the surface, the steeper inner one buys real clearance
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
        wids = contact_worldid.long()
        valid = (active & is_obs & (contact_dist < 0.0)
                 & (wids >= 0) & (wids < num_samples))

        # unconditional masked scatter: invalid slots land on index 0 with
        # weight 0, so no data-dependent branch syncs the host
        counts = torch.zeros(num_samples, device=device)
        counts.scatter_add_(0, wids.clamp(0, num_samples - 1),
                            valid.to(counts.dtype))
        return -counts

    def get_reward(config, env, ref):
        qpos_ref, qvel_ref = ref[0], ref[1]
        qpos_sim = wp.to_torch(env.data_wp.qpos)
        qvel_sim = wp.to_torch(env.data_wp.qvel)
        device = config.device

        # mjwp._diff_qpos's humanoid branch, inlined with the sync-free
        # quat_sub and studio's own weights
        qpos_diff = torch.cat(
            [
                qpos_sim[:, :3] - qpos_ref[:3],
                quat_sub(qpos_sim[:, 3:7], qpos_ref[3:7]),
                qpos_sim[:, 7:] - qpos_ref[7:],
            ],
            dim=1,
        )
        if device not in weight_cache:
            weight_cache[device] = build_qpos_weight(mj_model, params, device)
            arm_idx_cache[device] = arm_idx_cpu.to(device)
        qpos_dist = torch.norm(qpos_diff * weight_cache[device], p=2, dim=1)
        qvel_dist = torch.norm(qvel_sim - qvel_ref, p=2, dim=1)
        qpos_rew = -qpos_dist
        qvel_rew = -config.vel_rew_scale * qvel_dist

        # base drift, quadratic and OUTSIDE the tracking norm: the norm's
        # sqrt compresses base error whenever an arm deviates to avoid,
        # making drift the cheapest error to trade away
        base_pos_pen = (qpos_diff[:, :3] ** 2).sum(dim=1)
        base_rot_pen = (qpos_diff[:, 3:6] ** 2).sum(dim=1)
        # catches a forward trip while still momentum, not displacement
        base_vel_pen = torch.norm(qvel_sim[:, :2] - qvel_ref[:2], p=2, dim=1)
        # damps the press-shove-press oscillation where wrist tracking and
        # contact/SDF forces fight at the table
        arm_idx = arm_idx_cache[device]
        arm_vel_pen = torch.norm(qvel_sim[:, arm_idx] - qvel_ref[arm_idx],
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
