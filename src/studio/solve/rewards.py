"""Studio's reward stack on top of SPIDER's get_reward.

Two corrections to SPIDER's generic bimanual reward: a per-block
weighted velocity term (its flat L2 over all nv lets the 35 robot dims
swamp the 6 object dims), and a face-relative contact term targeting the
box where it ACTUALLY is rather than SPIDER's baked reference points.
`make_reward_fns` is the fused stack solves run.
"""

from __future__ import annotations

import torch
import warp as wp

from spider.config import Config
from spider.simulators.mjwp import (
    _weight_diff_qpos,
    get_reward as _spider_get_reward,
    get_terminal_reward as _spider_get_terminal_reward,
)

from .quat import quat_sub

# per qvel block; humanoid_object nv layout is [0:3] base lin, [3:6] base
# ang, [6:-6] joints, [-6:-3] object lin, [-3:] object ang
# the grasp term: a hand geom counts as touching the object within this
# (soft contacts sit a few mm deep), and a hand with this many touching
# geoms (palm + a finger's worth of capsules) has grasped fully
GRASP_TOUCH = 0.002        # m
GRASP_CONTACT_CAP = 6.0
GRASP_PEN_TOL = 0.005      # m: hand-object penetration priced from here
                           # (soft contacts sit a few mm deep when
                           # pressing; deeper is the hand digging in)
# the support term: a body geom the reference rests the object on has
# supported it once this many contact points bear on it
SUPPORT_CONTACT_CAP = 2.0

VEL_WEIGHTS = {
    "base_lin": 0.06,
    "base_ang": 0.02,
    "joint": 0.02,
    "obj_lin": 0.15,
    "obj_ang": 0.015,
}


def make_weighted_vel_reward_fns(config: Config):
    """(get_reward, get_terminal_reward) with the weighted qvel term.

    Zeroes config.vel_rew_scale so SPIDER's flat full-nv term stays
    inert; the stashed value scales the weighted term instead."""
    vel_scale = float(config.vel_rew_scale)
    config.vel_rew_scale = 0.0

    # ref may carry studio's extra channels; spider unpacks exactly five
    def base_reward(config, env, ref):
        return _spider_get_reward(config, env, ref[:5])

    def base_terminal(config, env, ref):
        return _spider_get_terminal_reward(config, env, ref[:5])

    if vel_scale == 0.0:
        return base_reward, base_terminal

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
        rew, info = base_reward(config, env, ref)
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
    """Wrap the reward with a simulated-box face contact term.

    Replaces SPIDER's baked contact_pos_ref points: each palm targets the
    nearest point on its own grasp face of the box where it actually is
    (left palm -> +x_box, right -> -x_box, the emit_trial site order), so
    the grip point is optimized per sample even on a drifted box.
    contact_rew_scale scales the term (SPIDER only gates on it); press
    biases the target inside the face, edge_margin keeps it off the
    corners."""
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


def make_reward_fns(config: Config, box_half=None, grips=None,
                    self_pairs=None, grasp=None, support=None):
    """The whole reward stack fused into one torch.compile'd core.

    Wrapper-per-term composition launched ~50 tiny kernels per rollout
    sim step (~3.5 ms/step at 4096 worlds, against a ~12 ms physics
    step); this computes the same reward in one pure-tensor function over
    tensors gathered once per call. Terms, all optional:

    - qpos tracking (spider's `_weight_diff_qpos`, hoisted to build time)
      and the weighted qvel term; endpoint-shaped object tracking via
      ref[5], loop.py's per-step weight — a solve without that channel
      runs unshaped.
    - contact: the face term when box_half is given, else spider's baked
      palm track. Both scale with contact_rew_scale (spider only gates).
    - grips: (pocket, palm column, anchor) per holding hand — the
      SIMULATED object's anchor point pulled into that palm's calibrated
      grasp pocket, gated by the reference contact mask. A zero anchor
      targets the object center.
    - upright: penalty for tilting past / sinking below the reference's
      own posture, so a fall costs more than a pose error.
    - self_pairs: summed penetration depth over the template's ghost
      self-collision pairs, which exert no force.
    - grasp: (geom -> palm column, object-geom mask) — rewards each
      holding hand for the number of its geoms (palm box + finger
      capsules) actually touching the object, saturating at
      GRASP_CONTACT_CAP, gated by the reference contact mask, and
      (grasp_pen_rew_scale) penalises their penetration depth beyond
      GRASP_PEN_TOL, summed, ungated. The grip term above holds the
      object in the pocket; the contact count asks the fingers to close
      on it, which sampling alone finds late; the depth penalty keeps
      the palm from burying a member inside its box, where the box
      cages it and the object gets carried by penetration force.
    - support: (geom -> support group, number of groups) — rewards each
      body-support group (torso, hip, thigh, forearm... whatever the
      reference rests the object on) for its contacts with the object,
      saturating at SUPPORT_CONTACT_CAP, gated per group by ref[6],
      loop.py's per-step support windows. Multi-contact carries are
      held by these as much as by the hands.

    Info dicts are empty; nothing in studio reads them."""
    device = config.device
    if config.embodiment_type != "humanoid_object":
        # the inlined qpos-difference layout below is humanoid_object's
        raise ValueError(
            "fused reward only supports humanoid_object, got "
            f"{config.embodiment_type}"
        )

    # --- constants hoisted to build time --------------------------------
    vel_scale = float(config.vel_rew_scale)
    config.vel_rew_scale = 0.0  # spider's flat term stays inert
    w_vel = torch.full((config.nv,), VEL_WEIGHTS["joint"], device=device)
    w_vel[:3] = VEL_WEIGHTS["base_lin"]
    w_vel[3:6] = VEL_WEIGHTS["base_ang"]
    w_vel[-6:-3] = VEL_WEIGHTS["obj_lin"]
    w_vel[-3:] = VEL_WEIGHTS["obj_ang"]
    w_vel = w_vel * vel_scale

    w_track = _weight_diff_qpos(config)

    sites = list(config.contact_site_ids)
    contact_scale = float(config.contact_rew_scale)
    use_face = (box_half is not None and contact_scale != 0.0
                and len(sites) == 2)
    use_baked = (not use_face and contact_scale > 0.0 and len(sites) > 0)
    if use_face:
        config.contact_rew_scale = 0.0  # the face term replaces it
        hw, hd, hh = (float(v) for v in box_half)
        face_x = hw  # press = 0.0
        edge_margin = 0.02
        my = max(hd - edge_margin, 0.5 * hd)
        mz = max(hh - edge_margin, 0.5 * hh)
        face_sign = torch.tensor([1.0, -1.0], device=device)  # left, right

    grip_scale = float(getattr(config, "grip_rew_scale", 0.0))
    grips = [g for g in (grips or []) if g[0] is not None]
    use_grips = grip_scale != 0.0 and bool(grips)
    if use_grips:
        g_sites = torch.tensor([sites[g[1]] for g in grips],
                               device=device, dtype=torch.long)
        g_cols = torch.tensor([g[1] for g in grips],
                              device=device, dtype=torch.long)
        g_pockets = torch.tensor([g[0] for g in grips],
                                 device=device, dtype=torch.float32)
        g_anchors = torch.tensor(
            [g[2] if (g[2] is not None and any(g[2])) else (0.0, 0.0, 0.0)
             for g in grips],
            device=device, dtype=torch.float32)
        grips_anchored = bool(g_anchors.abs().sum() > 0)

    self_scale = float(getattr(config, "self_collision_rew_scale", 0.0))
    use_self = self_scale != 0.0 and bool(self_pairs)
    if use_self:
        self_keys = torch.tensor(
            sorted((min(a, b) << 16) + max(a, b) for a, b in self_pairs),
            dtype=torch.int64, device=device)

    grasp_scale = float(getattr(config, "grasp_rew_scale", 0.0))
    grasp_pen_scale = float(getattr(config, "grasp_pen_rew_scale", 0.0))
    use_grasp = ((grasp_scale != 0.0 or grasp_pen_scale != 0.0)
                 and grasp is not None and bool(grasp[1].any()))
    if use_grasp:
        geom_hand = torch.tensor(grasp[0], dtype=torch.long, device=device)
        geom_obj = torch.tensor(grasp[1], dtype=torch.bool, device=device)
        n_hand_cols = int(geom_hand.max().item()) + 1

    upright_scale = float(getattr(config, "upright_rew_scale", 0.0))
    use_upright = upright_scale != 0.0
    objw_one = torch.ones((), device=device)

    need_xmat = use_grips
    need_rot = use_face or (use_grips and grips_anchored)
    need_sites = use_face or use_baked or use_grips
    support_scale = float(getattr(config, "support_rew_scale", 0.0))
    use_support = (support_scale != 0.0 and support is not None
                   and int(support[1]) > 0)
    if use_support:
        geom_supp = torch.tensor(support[0], dtype=torch.long, device=device)
        n_supp = int(support[1])
        if not use_grasp:
            geom_obj = torch.tensor(grasp[1], dtype=torch.bool, device=device)
    supp_none = torch.zeros(1, device=device)

    need_contacts = use_self or use_grasp or use_support
    num_samples = int(config.num_samples)
    terminal_scale = float(config.terminal_rew_scale)

    # --- the fused core: pure tensor ops, one call per sim step ---------
    def core(qpos, qvel, site_xpos, site_xmat, cgeom, cdist, cdim, cwid,
             ref_qpos, ref_qvel, ref_contact, ref_cpos, ref_objw, ref_supp):
        # mjwp._diff_qpos's humanoid_object branch inlined with the
        # sync-free quat_sub (spider's branches per call, see .quat);
        # ref_objw shapes the OBJECT entries over time — full weight at
        # the interaction's endpoints, discounted through the transit
        qpos_diff = torch.cat(
            [
                qpos[:, :3] - ref_qpos[:3],
                quat_sub(qpos[:, 3:7], ref_qpos[3:7]),
                qpos[:, 7:-7] - ref_qpos[7:-7],
                (qpos[:, -7:-4] - ref_qpos[-7:-4]) * ref_objw,
                quat_sub(qpos[:, -4:], ref_qpos[-4:]) * ref_objw,
            ],
            dim=1,
        )
        rew = -torch.norm(qpos_diff * w_track, p=2, dim=1)
        rew = rew - torch.norm((qvel - ref_qvel) * w_vel, p=2, dim=1)

        if use_upright:
            # base z-axis's world-z component, from the wxyz base quat
            upz = 1.0 - 2.0 * (qpos[:, 4] ** 2 + qpos[:, 5] ** 2)
            upz_ref = 1.0 - 2.0 * (ref_qpos[4] ** 2 + ref_qpos[5] ** 2)
            pen = (upz_ref - upz).clamp(min=0.0) \
                + (ref_qpos[2] - qpos[:, 2]).clamp(min=0.0)
            rew = rew - upright_scale * pen

        if need_rot:
            rot = _quat_to_mat_wxyz(qpos[:, -4:])             # box -> world
        if use_face:
            box_p = qpos[:, -7:-4]
            palms = site_xpos[:, sites]                        # (N, 2, 3)
            pl = torch.einsum("nji,nkj->nki", rot,
                              palms - box_p.unsqueeze(1))
            tgt = torch.stack(
                [
                    (face_sign * face_x).expand_as(pl[..., 0]),
                    pl[..., 1].clamp(-my, my),
                    pl[..., 2].clamp(-mz, mz),
                ],
                dim=-1,
            )
            face_dist = torch.norm(pl - tgt, dim=-1)           # (N, 2)
            rew = rew - contact_scale * (
                face_dist * ref_contact.unsqueeze(0)).sum(dim=1)
        elif use_baked:
            # scaled, not just gated as in spider
            cdist_ref = torch.norm(site_xpos[:, sites] - ref_cpos, p=2,
                                   dim=-1)
            rew = rew - contact_scale * (
                cdist_ref * ref_contact.unsqueeze(0)).sum(dim=1)

        if use_grips:
            pw = site_xpos[:, g_sites] + torch.einsum(
                "ngij,gj->ngi", site_xmat[:, g_sites], g_pockets)
            tgt = qpos[:, -7:-4].unsqueeze(1)
            if grips_anchored:
                tgt = tgt + torch.einsum("nij,gj->ngi", rot, g_anchors)
            dist = torch.norm(pw - tgt, dim=-1)
            rew = rew - grip_scale * (dist * ref_contact[g_cols]).sum(dim=1)

        if use_self:
            key = (torch.minimum(cgeom[:, 0], cgeom[:, 1]) << 16) \
                + torch.maximum(cgeom[:, 0], cgeom[:, 1])
            valid = ((cdim > 0) & (cdist < 0.0) & torch.isin(key, self_keys)
                     & (cwid >= 0) & (cwid < num_samples))
            depth = torch.zeros(num_samples, device=cdist.device)
            depth.scatter_add_(0, cwid.clamp(0, num_samples - 1),
                               torch.where(valid, -cdist,
                                           torch.zeros_like(cdist)))
            rew = rew - self_scale * depth

        if use_grasp:
            # a contact is a grasp contact when one geom is a hand geom
            # and the other the object; count them per world and hand
            a, b = cgeom[:, 0], cgeom[:, 1]
            col = torch.where(geom_obj[b], geom_hand[a],
                              torch.where(geom_obj[a], geom_hand[b],
                                          torch.full_like(a, -1)))
            live = ((cdim > 0) & (col >= 0) & (cwid >= 0)
                    & (cwid < num_samples))
            wid = cwid.clamp(0, num_samples - 1)
            if grasp_scale != 0.0:
                valid = live & (cdist < GRASP_TOUCH)
                idx = wid * n_hand_cols + col.clamp(0, n_hand_cols - 1)
                counts = torch.zeros(num_samples * n_hand_cols,
                                     device=cdist.device)
                counts.scatter_add_(0, idx, valid.to(counts.dtype))
                counts = counts.view(num_samples, n_hand_cols)
                held = counts.clamp(max=GRASP_CONTACT_CAP) / GRASP_CONTACT_CAP
                gate = ref_contact[:n_hand_cols].unsqueeze(0)
                rew = rew + grasp_scale * (held * gate).sum(dim=1)
            if grasp_pen_scale != 0.0:
                deep = torch.where(live, (-cdist - GRASP_PEN_TOL).clamp(min=0.0),
                                   torch.zeros_like(cdist))
                depth = torch.zeros(num_samples, device=cdist.device)
                depth.scatter_add_(0, wid, deep)
                rew = rew - grasp_pen_scale * depth

        if use_support:
            # a support contact: one geom in a support group, the other
            # the object; count them per world and group
            a, b = cgeom[:, 0], cgeom[:, 1]
            grp = torch.where(geom_obj[b], geom_supp[a],
                              torch.where(geom_obj[a], geom_supp[b],
                                          torch.full_like(a, -1)))
            valid = ((cdim > 0) & (cdist < GRASP_TOUCH) & (grp >= 0)
                     & (cwid >= 0) & (cwid < num_samples))
            idx = cwid.clamp(0, num_samples - 1) * n_supp \
                + grp.clamp(0, n_supp - 1)
            counts = torch.zeros(num_samples * n_supp, device=cdist.device)
            counts.scatter_add_(0, idx, valid.to(counts.dtype))
            counts = counts.view(num_samples, n_supp)
            borne = counts.clamp(max=SUPPORT_CONTACT_CAP) / SUPPORT_CONTACT_CAP
            rew = rew + support_scale * (borne * ref_supp[:n_supp]
                                         .unsqueeze(0)).sum(dim=1)

        return rew

    if getattr(config, "use_torch_compile", False):
        core = torch.compile(core, dynamic=False)

    def gather(env, ref):
        data = env.data_wp
        site_xpos = wp.to_torch(data.site_xpos) if need_sites else None
        site_xmat = None
        if need_xmat:
            site_xmat = wp.to_torch(data.site_xmat)
            site_xmat = site_xmat.reshape(site_xmat.shape[0], -1, 3, 3)
        cgeom = cdist = cdim = cwid = None
        if need_contacts:
            cgeom = wp.to_torch(data.contact.geom).long()
            cdist = wp.to_torch(data.contact.dist)
            cdim = wp.to_torch(data.contact.dim)
            cwid = wp.to_torch(data.contact.worldid).long()
        return (wp.to_torch(data.qpos), wp.to_torch(data.qvel),
                site_xpos, site_xmat, cgeom, cdist, cdim, cwid,
                ref[0], ref[1], ref[3], ref[4],
                ref[5] if len(ref) > 5 else objw_one,
                ref[6] if len(ref) > 6 else supp_none)

    def get_reward(config, env, ref):
        return core(*gather(env, ref)), {}

    def get_terminal_reward(config, env, ref):
        rew, info = get_reward(config, env, ref)
        return terminal_scale * rew, info

    return get_reward, get_terminal_reward
