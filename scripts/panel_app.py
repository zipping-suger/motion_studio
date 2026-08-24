"""Interactive tuning panel: reconstruct + solve from viser buttons.

Pick a motion from raw_motion/ — it previews immediately — and pick the
downstream task. Button 1 reconstructs the scene with the params set in
the GUI, button 2 solves it: box_carry, ground_pick and pole run
studio.solve, the obstacle tasks (under_table, kick) run
studio.solve.mppi_loop. Then watch reference (transparent) vs solution
(solid) playback.

Every action works inside an ordinary studio run dir (runs/<name>/), so
list/view/promote all apply.

Runs in the solve venv, which has viser + mujoco + SPIDER: launch with
`studio panel`, or `studio view <run>` to open an existing run.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import trimesh

STUDIO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDIO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

from studio import manifest, recon, shim, solve, viz
from studio import tasks as studio_tasks
from studio.config import (PICK_GRASPS, PICK_SCENE_DEFAULTS, PICK_SHAPES,
                           POLE_OBJECTS, POLE_SCENE_DEFAULTS,
                           RAW_MOTION_DIR, RUNS_DIR, SCENE_DEFAULTS,
                           SOLVE_DEFAULTS, SOLVE_INT_KEYS, load_config,
                           task_dirs)
from studio.recon.grasp import (BOX_WIDTH_MAX, BOX_WIDTH_MIN,
                                HAND_SURFACE_OFFSET, detect_grasp)
from studio.recon import layout
from studio.recon import pick as recon_pick
from studio.recon import pole as recon_pole
from studio.recon.loader import (DOF_NAMES, KIM_LEFT_HAND_TIP,
                                 KIM_RIGHT_HAND_TIP, load_kimodo_npz)
from studio.recon.signal import smooth
from studio.tasks import kick_params, under_table_params

import mujoco
import viser
from viser.extras import ViserUrdf
import yourdfpy

CFG = load_config()
# the BrainCo-hand model when its assets are installed, else the handless
# 29-DoF URDF. The motion drives the same 29 body DOFs either way; finger
# joints follow a trial's qpos and hold rest in raw previews.
G1_URDF = str(recon.assets.G1_BRAINCO_URDF
              if recon.assets.brainco_available() else recon.assets.G1_URDF)
MESHES_DIR = str(recon.assets.MESHES_DIR)

# the obstacle tasks the panel drives and the params it exposes; the full
# sets live in the *_params modules and everything else stays default
OBSTACLE_TASKS = {
    "under_table": {
        "params": under_table_params,
        "suffix": "ut",              # run name: <clip>_<suffix>
        "estimate": "1. Estimate table",
        "scene_keys": ("scene_seed", "hard_collision", "arm_conflict",
                       "underside_drop", "max_ref_table_pen", "yaw_range",
                       "table_width"),
        "solve_keys": ("num_samples", "max_num_iterations", "temperature",
                       "horizon", "sdf_weight", "collision_margin",
                       "collision_inner_scale", "collision_count_weight",
                       "stability_weight", "leg_noise_scale"),
    },
    "kick": {
        "params": kick_params,
        "suffix": "kick",
        "estimate": "1. Place kick obstacle",
        "scene_keys": ("scene_seed", "hard_collision", "path_frac",
                       "max_ref_pen", "box_depth", "box_width",
                       "min_box_height"),
        "solve_keys": ("num_samples", "max_num_iterations", "temperature",
                       "horizon", "sdf_weight", "collision_margin",
                       "collision_inner_scale", "collision_count_weight",
                       "stability_weight", "leg_noise_scale", "qw_legs"),
    },
}
# first primitive (slab / kick box) darker, table legs lighter, mirroring
# the injected scene rgba
OBSTACLE_COLORS = [(140, 89, 51), (115, 71, 38)]


# lh/rh palms plus, on the BrainCo model, the per-finger-segment capsules
HAND_GEOM_RE = re.compile(r"^[lr]h(_|$)")


def hand_collision_geoms(scene_xml, qpos_traj):
    """Names, meshes and world poses (via FK) of the full hand collision
    set — palm box/hull + finger-segment capsules. This is the actual
    contact model the solve grasps with, so the overlay must show all of
    it, not just the palms."""
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    geoms = [(n, g) for g in range(m.ngeom)
             if HAND_GEOM_RE.match(
                 n := mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    meshes = []
    for name, gid in geoms:
        if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            side = "left" if name[0] == "l" else "right"
            hand = trimesh.load(f"{MESHES_DIR}/{side}_rubber_hand.STL")
            hull = trimesh.convex.convex_hull(hand)
            mid = m.geom_dataid[gid]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.mesh_quat[mid])
            hull.vertices = (hull.vertices - m.mesh_pos[mid]) @ R.reshape(3, 3)
            meshes.append(hull)
        elif m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            # the BrainCo template's lh/rh palm boxes
            meshes.append(trimesh.creation.box(2 * m.geom_size[gid]))
        else:
            radius, half_len = m.geom_size[gid][0], m.geom_size[gid][1]
            meshes.append(trimesh.creation.capsule(radius=radius,
                                                   height=2 * half_len))
    poses = np.zeros((len(qpos_traj), len(geoms), 7))
    quat = np.zeros(4)
    for t, qp in enumerate(qpos_traj):
        d.qpos[:] = qp
        mujoco.mj_forward(m, d)
        for k, (_n, gid) in enumerate(geoms):
            poses[t, k, :3] = d.geom_xpos[gid]
            mujoco.mju_mat2Quat(quat, d.geom_xmat[gid].ravel())
            poses[t, k, 3:] = quat
    return [n for n, _ in geoms], meshes, poses


def object_collision_geoms(scene_xml):
    """The object's PHYSICAL model: every collision geom on the free
    object body — the handle capsule + base primitive of a pole trial,
    the (squeeze-widened) box of box_carry, the pick object's primitive
    — as (names, meshes, offsets), offsets being each geom's static
    body-frame (pos, wxyz quat). The OMOMO visual mesh is excluded: the
    point of the overlay is seeing exactly what physics can collide
    with, next to the visual it diverges from (legs, hooks and shades
    have no collision geom at all)."""
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "largebox")
    names, meshes, offsets = [], [], []
    if bid < 0:
        return names, meshes, offsets
    builders = {
        mujoco.mjtGeom.mjGEOM_BOX:
            lambda s: trimesh.creation.box(2 * s),
        mujoco.mjtGeom.mjGEOM_CAPSULE:
            lambda s: trimesh.creation.capsule(radius=s[0], height=2 * s[1]),
        mujoco.mjtGeom.mjGEOM_CYLINDER:
            lambda s: trimesh.creation.cylinder(radius=s[0], height=2 * s[1]),
        mujoco.mjtGeom.mjGEOM_SPHERE:
            lambda s: trimesh.creation.icosphere(radius=s[0]),
    }
    for g in range(m.ngeom):
        build = builders.get(m.geom_type[g])
        if m.geom_bodyid[g] != bid or build is None:
            continue  # the visual mesh, or an unknown primitive
        names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                     or f"geom{g}")
        meshes.append(build(m.geom_size[g]))
        offsets.append((m.geom_pos[g].copy(), m.geom_quat[g].copy()))
    return names, meshes, offsets


def list_sources():
    # inputs all come from raw_motion/: demo saves at its top level, batch
    # clips in subdirs. Keyed by relative path so equal stems stay distinct.
    RAW_MOTION_DIR.mkdir(exist_ok=True)
    return {str(f.relative_to(RAW_MOTION_DIR).with_suffix("")): f
            for f in sorted(RAW_MOTION_DIR.rglob("*.npz"))}


class Panel:
    def __init__(self, server: viser.ViserServer):
        self.server = server
        self.run_dir = None
        self.task_dir = None
        self.ref_qpos = None
        self.res_qpos = None
        self.hand_poses = None
        self.hand_handles = []
        self.obj_col_handles = []   # object PHYSICAL-model overlay
        self.obj_col_offsets = []   # matching body-frame (pos, quat)
        self.scene_handles = []
        self.busy = False
        # published ref/res layout: raw previews use the 36-wide body
        # contract, trial files the model layout (studio.recon.layout).
        # q_addr maps the 29 body joints out of a model-layout frame;
        # joint_addr maps EVERY URDF joint so trial playback shows the
        # grasp closure the qpos carries.
        self.q_addr = None
        self.joint_addr = None
        self.ref_has_box = False
        self.res_has_box = False
        # contact-window preview state (only while previewing a raw motion)
        self.auto_window = None     # (pick, release) as detected
        self.ghost = None           # {"pick","rel"} — publish LAST
        self.ghost_box = None
        self._mid = self._yaw = self._gap = None
        self._suppress_window_cb = False
        self._suppress_task_cb = False

        server.scene.add_grid("/ground", width=6, height=6)
        self.ref_box = None
        self.res_box = None
        self.ref_base, self.ref_vis, self.order = self._add_robot(
            "ref_robot", alpha=0.35)
        self.res_base = self.res_vis = None
        self.ref_base.visible = False

        self._build_gui()

    def _add_robot(self, name, alpha=None):
        urdf_model = yourdfpy.URDF.load(G1_URDF, load_collision_meshes=False)
        base = self.server.scene.add_frame(f"/{name}", show_axes=False)
        kwargs = {}
        if alpha is not None:
            kwargs["mesh_color_override"] = (0.5, 0.5, 0.8, alpha)
        vis = ViserUrdf(self.server, urdf_model, root_node_name=f"/{name}",
                        **kwargs)
        # qpos index per actuated joint; None for the BrainCo fingers,
        # which the body-contract motion does not drive and _cfg rests
        self.joint_names = list(urdf_model.actuated_joint_names)
        order = [DOF_NAMES.index(n) if n in DOF_NAMES else None
                 for n in self.joint_names]
        rest = np.zeros(len(order))
        for k, n in enumerate(self.joint_names):
            lim = urdf_model.joint_map[n].limit
            if lim is not None:  # some finger tips pin at 1.0 rad
                rest[k] = np.clip(0.0, lim.lower, lim.upper)
        return base, vis, (order, rest)

    def _cfg(self, frame):
        """update_cfg vector from a qpos frame in either layout. Body DOFs
        come from the motion; the BrainCo finger joints follow the frame
        when the trial's model carries them — that is the grasp closure —
        and hold rest for raw body-contract previews."""
        order, rest = self.order
        if frame.shape[-1] != layout.BODY_NQ and self.joint_addr is not None:
            return np.array([rest[k] if a is None else frame[a]
                             for k, a in enumerate(self.joint_addr)])
        joints = self._body29(frame)
        return np.array([rest[k] if i is None else joints[i]
                         for k, i in enumerate(order)])

    def _body29(self, frame):
        """The 29 body joint values from a qpos frame in either layout."""
        if frame.shape[-1] == layout.BODY_NQ:
            return frame[7:36]
        return frame[self.q_addr]

    def _load_q_addr(self):
        """Joint addresses for this trial's model layout: the 29 body
        joints (_body29) and every URDF joint by name (_cfg, which is how
        the finger closure in a trial's qpos reaches the rendered hands)."""
        m = mujoco.MjModel.from_xml_path(str(self.task_dir / "scene.xml"))
        self.q_addr = layout.body_addr(m)
        addr = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            addr.append(int(m.jnt_qposadr[jid]) if jid >= 0 else None)
        self.joint_addr = addr

    # ------------------------------------------------------------- GUI --
    def _build_gui(self):
        gui = self.server.gui
        with gui.add_folder(f"Motion ({RAW_MOTION_DIR.name}/)"):
            self.sources = list_sources()
            options = list(self.sources) or ["<none found>"]
            self.gui_source = gui.add_dropdown("source", options,
                                               initial_value=options[0])
            self.gui_task = gui.add_dropdown("task", tuple(studio_tasks.names()),
                                             initial_value=studio_tasks.DEFAULT)
            refresh = gui.add_button("refresh sources")

            @refresh.on_click
            def _(_):
                self.sources = list_sources()
                opts = list(self.sources) or ["<none found>"]
                self.gui_source.options = opts

        # same controls as the demo's "Scene recon" folder
        self.folder_box_scene = gui.add_folder("Scene reconstruction")
        with self.folder_box_scene:
            self.scene_widgets = viz.scene_param_widgets(gui, SCENE_DEFAULTS)
            self.gui_held = gui.add_checkbox("allow held start", False)
            # auto-detected on preview, freely editable; the ghost box
            # shows the object trajectory the window implies
            self.gui_cstart = gui.add_number("contact start", 0, min=0, step=1)
            self.gui_cend = gui.add_number("contact end", 0, min=0, step=1)
            self.btn_auto = gui.add_button("auto window")
            self.btn_recon = gui.add_button("1. Reconstruct scene")
            self.md_recon = gui.add_markdown("*no reconstruction yet*")

        self.folder_box_solve = gui.add_folder("SBMPC solve")
        with self.folder_box_solve:
            self.solve_widgets = {}
            for key, val in SOLVE_DEFAULTS.items():
                if key in SOLVE_INT_KEYS:
                    self.solve_widgets[key] = gui.add_number(
                        key, int(val), min=1, step=max(1, int(val) // 16))
                else:
                    self.solve_widgets[key] = gui.add_number(
                        key, float(val), min=0.0, step=0.05)
            self.btn_solve = gui.add_button("2. Solve SBMPC")
            self.btn_solve.disabled = True
            self.solve_progress = gui.add_progress_bar(0.0, visible=False)
            self.md_solve = gui.add_markdown("*reconstruct first*")

        # ground_pick: object params + box_carry's SBMPC solve controls
        self.folder_pick = gui.add_folder("ground_pick task", visible=False)
        with self.folder_pick:
            self.pick_scene_widgets = {}
            choices = {"shape": PICK_SHAPES, "grasp": PICK_GRASPS}
            for key, val in PICK_SCENE_DEFAULTS.items():
                if key in choices:
                    self.pick_scene_widgets[key] = gui.add_dropdown(
                        key, choices[key], initial_value=val)
                else:
                    self.pick_scene_widgets[key] = gui.add_number(
                        key, float(val), min=0.0, step=0.01)
            self.btn_pick_recon = gui.add_button("1. Reconstruct pick scene")
            self.md_pick_recon = gui.add_markdown("*no reconstruction yet*")
            self.pick_solve_widgets = {}
            for key, val in SOLVE_DEFAULTS.items():
                if key in SOLVE_INT_KEYS:
                    self.pick_solve_widgets[key] = gui.add_number(
                        key, int(val), min=1, step=max(1, int(val) // 16))
                else:
                    self.pick_solve_widgets[key] = gui.add_number(
                        key, float(val), min=0.0, step=0.05)
            self.btn_pick_solve = gui.add_button("2. Solve SBMPC")
            self.btn_pick_solve.disabled = True
            self.pick_progress = gui.add_progress_bar(0.0, visible=False)
            self.md_pick_solve = gui.add_markdown("*reconstruct first*")

        # pole: object/mode params + the same SBMPC solve controls
        self.folder_pole = gui.add_folder("pole task", visible=False)
        with self.folder_pole:
            self.pole_scene_widgets = {}
            choices = {"object": POLE_OBJECTS, "grasp": PICK_GRASPS}
            for key, val in POLE_SCENE_DEFAULTS.items():
                if key in choices:
                    self.pole_scene_widgets[key] = gui.add_dropdown(
                        key, choices[key], initial_value=val)
                elif isinstance(val, str):
                    # free-form: auto | off | S-E frames
                    self.pole_scene_widgets[key] = gui.add_text(
                        key, initial_value=val)
                else:
                    self.pole_scene_widgets[key] = gui.add_number(
                        key, float(val), min=0.0, step=0.01)
            self.btn_pole_recon = gui.add_button("1. Reconstruct pole scene")
            self.md_pole_recon = gui.add_markdown("*no reconstruction yet*")
            self.pole_solve_widgets = {}
            for key, val in SOLVE_DEFAULTS.items():
                if key in SOLVE_INT_KEYS:
                    self.pole_solve_widgets[key] = gui.add_number(
                        key, int(val), min=1, step=max(1, int(val) // 16))
                else:
                    self.pole_solve_widgets[key] = gui.add_number(
                        key, float(val), min=0.0, step=0.05)
            self.btn_pole_solve = gui.add_button("2. Solve SBMPC")
            self.btn_pole_solve.disabled = True
            self.pole_progress = gui.add_progress_bar(0.0, visible=False)
            self.md_pole_solve = gui.add_markdown("*reconstruct first*")

        self.obs = {}
        for tname, tcfg in OBSTACLE_TASKS.items():
            p = tcfg["params"]
            folder = gui.add_folder(f"{tname} task", visible=False)
            h = {"folder": folder, "scene_widgets": {}, "solve_widgets": {}}
            with folder:
                for key in tcfg["scene_keys"]:
                    val = p.SCENE_DEFAULTS[key]
                    if isinstance(val, bool):
                        h["scene_widgets"][key] = gui.add_checkbox(key, val)
                    elif key == "scene_seed":
                        h["scene_widgets"][key] = gui.add_number(
                            key, int(val), min=0, step=1)
                    else:
                        h["scene_widgets"][key] = gui.add_number(
                            key, float(val), min=0.0, step=0.01)
                h["btn_recon"] = gui.add_button(tcfg["estimate"])
                h["md_recon"] = gui.add_markdown("*no scene yet*")
                for key in tcfg["solve_keys"]:
                    val = p.SOLVE_DEFAULTS[key]
                    if key in p.SOLVE_INT_KEYS:
                        h["solve_widgets"][key] = gui.add_number(
                            key, int(val), min=1, step=max(1, int(val) // 16))
                    else:
                        h["solve_widgets"][key] = gui.add_number(
                            key, float(val), min=0.0,
                            step=1.0 if val >= 10 else 0.05)
                h["btn_solve"] = gui.add_button("2. Solve MPPI")
                h["btn_solve"].disabled = True
                h["progress"] = gui.add_progress_bar(0.0, visible=False)
                h["md_solve"] = gui.add_markdown("*build the scene first*")
            self.obs[tname] = h

        with gui.add_folder("Playback"):
            self.gui_frame = gui.add_slider("frame", 0, 99, 1, 0)
            self._frame_order = self.gui_frame.order
            self.gui_play = gui.add_checkbox("play", True)
            self.gui_speed = gui.add_slider("speed", 0.1, 2.0, 0.1, 1.0)
            # stacked reference/result playback is hard to read: hide
            # either side, or the yellow hand geoms, to see the other
            self.gui_show_ref = gui.add_checkbox("show reference", True)
            self.gui_show_res = gui.add_checkbox("show result", True)
            self.gui_show_hands = gui.add_checkbox("show hand collision",
                                                   True)
            # the object's PHYSICAL model: handle capsule + base
            # primitive, since the visual mesh has no collision
            self.gui_show_objcol = gui.add_checkbox("show object collision",
                                                    False)

        for w in (self.gui_show_ref, self.gui_show_res, self.gui_show_hands,
                  self.gui_show_objcol):
            w.on_update(lambda _e: self._apply_visibility())

        @self.gui_source.on_update
        def _(_):
            self._spawn(self.do_preview)

        @self.gui_task.on_update
        def _(_):
            if self._suppress_task_cb or self.busy:
                return
            self._apply_task_mode()
            self._spawn(self.do_preview)  # reset the scene for the new task

        @self.gui_held.on_update
        def _(_):
            if self.ghost is not None:  # re-detect under the new mode
                self._spawn(self.do_preview)

        @self.btn_auto.on_click
        def _(_):
            # apply the detected window (preview defaults to the full clip)
            if self.auto_window is None or self.busy:
                return
            self._suppress_window_cb = True
            self.gui_cstart.value = self.auto_window[0]
            self.gui_cend.value = self.auto_window[1]
            self._suppress_window_cb = False
            self._rebuild_ghost()

        @self.gui_cstart.on_update
        def _(_):
            if not self._suppress_window_cb:
                self._rebuild_ghost()

        @self.gui_cend.on_update
        def _(_):
            if not self._suppress_window_cb:
                self._rebuild_ghost()

        @self.btn_recon.on_click
        def _(_):
            self._spawn(self.do_reconstruct)

        @self.btn_solve.on_click
        def _(_):
            self._spawn(self.do_solve)

        @self.btn_pick_recon.on_click
        def _(_):
            self._spawn(self.do_reconstruct_pick)

        @self.btn_pick_solve.on_click
        def _(_):
            self._spawn(self.do_solve_pick)

        @self.btn_pole_recon.on_click
        def _(_):
            self._spawn(self.do_reconstruct_pole)

        @self.btn_pole_solve.on_click
        def _(_):
            self._spawn(self.do_solve_pole)

        for tname, h in self.obs.items():
            # bind the task name per closure, not per loop variable
            h["btn_recon"].on_click(
                lambda _e, tn=tname: self._spawn(
                    lambda: self.do_reconstruct_obs(tn)))
            h["btn_solve"].on_click(
                lambda _e, tn=tname: self._spawn(
                    lambda: self.do_solve_obs(tn)))

        self._spawn(self.do_preview)  # preview the initial selection

    def _apply_task_mode(self):
        active = self.gui_task.value
        box = active == "box_carry"
        self.folder_box_scene.visible = box
        self.folder_box_solve.visible = box
        self.folder_pick.visible = active == "ground_pick"
        self.folder_pole.visible = active == "pole"
        for tname, h in self.obs.items():
            h["folder"].visible = tname == active

    def _apply_visibility(self):
        """Reference / result / hand-overlay visibility from the playback
        toggles. Called on toggle and at the end of every load, so a
        hidden side stays hidden across reconstructs and solves."""
        try:
            self.ref_base.visible = (self.gui_show_ref.value
                                     and self.ref_qpos is not None)
            if self.ref_box is not None:
                self.ref_box.visible = self.gui_show_ref.value
            show_res = self.gui_show_res.value and self.res_qpos is not None
            if self.res_base is not None:
                self.res_base.visible = show_res
            if self.res_box is not None:
                self.res_box.visible = show_res
            show_hands = (self.gui_show_hands.value
                          and self.res_qpos is not None)
            for h in self.hand_handles:
                h.visible = show_hands
            for h in self.obj_col_handles:
                h.visible = self.gui_show_objcol.value
        except RuntimeError:
            pass  # a worker replaced a handle mid-toggle; the load that
            # replaced it calls this again with the fresh handles

    def _reset_frame_slider(self, T: int):
        """Slider bounds are immutable in viser — recreate the frame slider
        per motion so playback shows real frame numbers."""
        old = self.gui_frame
        self.gui_frame = self.server.gui.add_slider(
            "frame", 0, max(T - 1, 1), 1, 0, order=self._frame_order)
        old.remove()

    def _spawn(self, fn):
        if self.busy:
            return
        threading.Thread(target=fn, daemon=True).start()

    def _set_busy(self, busy: bool):
        self.busy = busy
        self.btn_recon.disabled = busy
        self.gui_task.disabled = busy
        self.btn_solve.disabled = busy or self.task_dir is None
        self.btn_pick_recon.disabled = busy
        self.btn_pick_solve.disabled = busy or self.task_dir is None
        self.btn_pole_recon.disabled = busy
        self.btn_pole_solve.disabled = busy or self.task_dir is None
        for h in self.obs.values():
            h["btn_recon"].disabled = busy
            h["btn_solve"].disabled = busy or self.task_dir is None

    # --------------------------------------------------------- preview --
    def do_preview(self):
        """Play the raw kinematic motion (robot only) on source selection."""
        source = self.sources.get(self.gui_source.value)
        if source is None:
            return
        self._set_busy(True)
        try:
            qpos, meta = load_kimodo_npz(source, use_smooth_pos=False)  # (T,36)
            self.ref_qpos = None
            self.res_qpos = None
            self.task_dir = None  # solve needs a fresh reconstruction
            self._clear_ghost()
            for h in (self.scene_handles + self.hand_handles
                      + self.obj_col_handles):
                h.remove()
            self.scene_handles, self.hand_handles = [], []
            self.obj_col_handles, self.obj_col_offsets = [], []
            if self.res_base is not None:
                self.res_base.visible = False
            if self.res_box is not None:
                self.res_box.remove()
                self.res_box = None
            if self.ref_box is not None:
                self.ref_box.remove()
                self.ref_box = None
            self._reset_frame_slider(len(qpos))
            self.ref_has_box = False  # raw preview: body contract, no box
            self.ref_qpos = qpos  # publish LAST
            self._apply_visibility()

            if self.gui_task.value in self.obs:
                # no ghost box / contact window in obstacle modes: the
                # scene is the estimated obstacle, drawn by button 1
                h = self.obs[self.gui_task.value]
                self._mid = self._yaw = self._gap = None
                self.auto_window = None
                h["md_recon"].content = (
                    f"*previewing `{source.stem}` ({len(qpos)} frames) — "
                    "set a seed and build the scene*")
                h["md_solve"].content = "*build the scene first*"
                return

            if self.gui_task.value == "ground_pick":
                # no ghost box either: the object spawns where the picking
                # hand bottoms out, which the detector reports here
                self._mid = self._yaw = self._gap = None
                self.auto_window = None
                try:
                    info = recon_pick.detect_pick(meta)
                    detail = (f"{info.hand} hand pick f{info.pick_frame} "
                              f"reach {info.reach_height:.2f} "
                              f"lift {info.lift_height:.2f} "
                              f"flags [{','.join(info.quality_flags) or '-'}]")
                except Exception as e:
                    detail = f"pick detection failed: {e}"
                self.md_pick_recon.content = (
                    f"*previewing `{source.stem}` ({len(qpos)} frames) — "
                    f"{detail}*")
                self.md_pick_solve.content = "*reconstruct first*"
                return

            if self.gui_task.value == "pole":
                # no ghost box: the pole spawns under the holding hands'
                # grasp pockets, which the detector reports here
                self._mid = self._yaw = self._gap = None
                self.auto_window = None
                try:
                    info = recon_pole.detect_pole_hold(meta, qpos)
                    detail = (f"{'+'.join(info.hands)} hold "
                              f"f{info.grab_frame}-f{info.end_frame} "
                              f"transport {info.transport:.2f} "
                              f"flags [{','.join(info.quality_flags) or '-'}]")
                except Exception as e:
                    detail = f"hold detection failed: {e}"
                self.md_pole_recon.content = (
                    f"*previewing `{source.stem}` ({len(qpos)} frames) — "
                    f"{detail}*")
                self.md_pole_solve.content = "*reconstruct first*"
                return

            # detect the window + precompute the hand-midpoint track for
            # the ghost box, mirroring build_object_trajectory
            grasp = detect_grasp(meta, qpos, {},
                                 allow_held_start=self.gui_held.value)
            jp = meta["joint_positions"]
            lh, rh = jp[:, KIM_LEFT_HAND_TIP], jp[:, KIM_RIGHT_HAND_TIP]
            mid = 0.5 * (lh + rh)
            mid = np.stack([smooth(mid[:, i], 5) for i in range(3)], axis=1)
            axis = lh - rh
            yaw = smooth(np.unwrap(np.arctan2(axis[:, 1], axis[:, 0])), 9)
            self._gap = np.linalg.norm(lh - rh, axis=1)
            self._mid, self._yaw = mid, yaw
            # default is the full clip; detected is one click away
            self.auto_window = (grasp.pick_frame, grasp.release_frame)
            T = len(qpos)
            self._suppress_window_cb = True
            self.gui_cstart.value = 0
            self.gui_cend.value = T - 1
            self._suppress_window_cb = False
            self._rebuild_ghost()
            self.md_recon.content = (
                f"*previewing `{source.stem}` ({T} frames) — contact window "
                f"defaults to the full clip; detected "
                f"f{grasp.pick_frame}–f{grasp.release_frame} "
                f"(flags: {','.join(grasp.quality_flags) or '-'}, apply "
                "with 'auto window'). The orange ghost box shows the scene "
                "the window implies*")
            self.md_solve.content = "*reconstruct first*"
        except Exception as e:
            self.md_recon.content = f"**preview failed**: {e}"
            self.md_pick_recon.content = f"**preview failed**: {e}"
            for h in self.obs.values():
                h["md_recon"].content = f"**preview failed**: {e}"
        finally:
            self._set_busy(False)

    def _clear_ghost(self):
        self.ghost = None  # unpublish before touching the handle
        if self.ghost_box is not None:
            self.ghost_box.remove()
            self.ghost_box = None

    def _rebuild_ghost(self):
        """(Re)create the ghost box for the current contact window: width
        from the window's median hand gap, pose riding the hand midpoint —
        exactly the object trajectory reconstruction will build."""
        if self._mid is None or self.ref_qpos is None:
            return
        T = len(self._mid)
        pick = int(np.clip(self.gui_cstart.value, 0, T - 2))
        rel = int(np.clip(self.gui_cend.value, pick + 1, T - 1))
        self._clear_ghost()
        width = float(np.clip(
            np.median(self._gap[pick:rel + 1]) - 2 * HAND_SURFACE_OFFSET,
            BOX_WIDTH_MIN, BOX_WIDTH_MAX))
        mesh = trimesh.creation.box(
            (width, float(self.scene_widgets["box_depth"].value),
             float(self.scene_widgets["box_height"].value)))
        self.ghost_box = self.server.scene.add_mesh_simple(
            "/ghost_box", mesh.vertices, mesh.faces,
            color=(240, 160, 40), opacity=0.35,
            position=self._mid[pick], wxyz=(1.0, 0, 0, 0))
        self.ghost = {"pick": pick, "rel": rel}  # publish LAST

    # ----------------------------------------------------- reconstruct --
    def do_reconstruct(self):
        source = self.sources.get(self.gui_source.value)
        if source is None:
            return
        self._set_busy(True)
        self.md_recon.content = "**reconstructing ...** (takes ~10-20 s)"
        try:
            name = shim.sanitize(source.stem)
            run_dir = RUNS_DIR / name
            run_dir.mkdir(parents=True, exist_ok=True)
            motion_dir = shim.shim_motion_npz(source, run_dir)
            npz = motion_dir / f"{name}_00.npz"

            scene_params = {k: w.value for k, w in self.scene_widgets.items()}
            if self.gui_held.value:
                scene_params["allow_held_start"] = True
            pick = int(self.gui_cstart.value)
            rel = int(self.gui_cend.value)
            # only force the window when the user moved it off the
            # detected values: the detector's window end differs from
            # release, so a no-op override would still shift the width
            override = (self.auto_window is not None
                        and (pick, rel) != self.auto_window)
            if override:
                scene_params["contact_window"] = [pick, rel]
            try:
                task_dir, _, line = recon.reconstruct(
                    npz, run_dir / "outputs", f"{name}_00",
                    {k: v for k, v in scene_params.items()
                     if k in SCENE_DEFAULTS},
                    allow_held_start=self.gui_held.value,
                    pick=pick if override else None,
                    release=rel if override else None)
            except Exception as e:
                self.md_recon.content = f"**reconstruction crashed**: {e}"
                self.task_dir = None
                return
            if task_dir is None:
                self.md_recon.content = f"**failed/skipped**\n```\n{line}\n```"
                self.task_dir = None
                return
            self.run_dir, self.task_dir = run_dir, task_dir
            manifest.update(run_dir, {"name": name, "source": str(source),
                                      "panel_scene_params": scene_params})
            self._load_reference()
            self.md_recon.content = f"```\n{line.split(' -> ')[0]}\n```"
            self.md_solve.content = "*ready to solve*"
        finally:
            self._set_busy(False)

    def _load_reference(self):
        self._clear_ghost()  # the real reconstructed box replaces it
        info = viz.read_info(self.task_dir)
        ref_qpos = np.load(
            self.task_dir / "0/trajectory_kinematic.npz")["qpos"]
        # unpublish first: tick() runs in the main thread and must not
        # touch handles while they are being rebuilt
        self.ref_qpos = None
        self.res_qpos = None
        for h in (self.scene_handles + self.hand_handles
                  + self.obj_col_handles):
            h.remove()
        self.scene_handles, self.hand_handles = [], []
        self.obj_col_handles, self.obj_col_offsets = [], []
        if self.res_base is not None:
            self.res_base.visible = False
        if self.res_box is not None:
            self.res_box.remove()
            self.res_box = None

        self.scene_handles = viz.add_scene(self.server, self.task_dir)

        if self.ref_box is not None:
            self.ref_box.remove()
        self._box_mesh = viz.object_mesh(info)
        self.ref_box = self.server.scene.add_mesh_simple(
            "/ref_box", self._box_mesh.vertices, self._box_mesh.faces,
            color=(200, 60, 60), opacity=0.4,
            position=ref_qpos[0, -7:-4], wxyz=ref_qpos[0, -4:])

        # the object's PHYSICAL model, toggled from the playback folder;
        # rides the shown pose in tick() (a free joint, so static geom
        # offsets compose with qpos[-7:] and no per-frame FK is needed)
        oc_names, oc_meshes, oc_offsets = object_collision_geoms(
            self.task_dir / "scene.xml")
        for k, name in enumerate(oc_names):
            self.obj_col_handles.append(self.server.scene.add_mesh_simple(
                f"/objcol_{name}", oc_meshes[k].vertices,
                oc_meshes[k].faces, color=(60, 120, 220), opacity=0.55,
                position=ref_qpos[0, -7:-4], wxyz=ref_qpos[0, -4:],
                visible=self.gui_show_objcol.value))
        self.obj_col_offsets = oc_offsets
        self._reset_frame_slider(len(ref_qpos))
        self._load_q_addr()
        self.ref_has_box = True  # model layout, box joint last
        self.ref_qpos = ref_qpos  # publish LAST: handles are complete now
        self._apply_visibility()

    # ---------------------------------------------- ground_pick recon --
    def do_reconstruct_pick(self):
        """Build a ground_pick trial. The trial keeps box_carry's contract
        (object free joint last), so reference/result rendering is the
        box path — only the object mesh differs."""
        source = self.sources.get(self.gui_source.value)
        if source is None:
            return
        self._set_busy(True)
        self.md_pick_recon.content = "**reconstructing ...** (takes ~10-20 s)"
        try:
            # own run name: a box_carry run of the same clip may coexist
            name = shim.sanitize(f"{source.stem}_gp")
            run_dir = RUNS_DIR / name
            run_dir.mkdir(parents=True, exist_ok=True)
            motion_dir = shim.shim_motion_npz(source, run_dir)
            npz = motion_dir / f"{name}_00.npz"

            scene_params = {k: w.value
                            for k, w in self.pick_scene_widgets.items()}
            task_obj = studio_tasks.load("ground_pick")
            try:
                task_dir, line = task_obj.reconstruct(
                    npz, run_dir / "outputs", f"{name}_00",
                    {"scene_params": scene_params})
            except Exception as e:
                self.md_pick_recon.content = f"**reconstruction crashed**: {e}"
                self.task_dir = None
                return
            if task_dir is None:
                self.md_pick_recon.content = (
                    f"**failed/skipped**\n```\n{line}\n```")
                self.task_dir = None
                return
            self.run_dir, self.task_dir = run_dir, task_dir
            manifest.update(run_dir, {
                "name": name, "source": str(source),
                "task_type": "ground_pick", "verdict": "recon",
                "panel_scene_params": scene_params})
            self._load_reference()
            self.md_pick_recon.content = f"```\n{line.split(' -> ')[0]}\n```"
            self.md_pick_solve.content = "*ready to solve*"
        finally:
            self._set_busy(False)

    # ---------------------------------------------------- pole recon --
    def do_reconstruct_pole(self):
        """Build a pole trial. The trial keeps box_carry's contract
        (object free joint last), so reference/result rendering is the
        box path — only the object mesh differs."""
        source = self.sources.get(self.gui_source.value)
        if source is None:
            return
        self._set_busy(True)
        self.md_pole_recon.content = "**reconstructing ...** (takes ~10-30 s)"
        try:
            # own run name: a box_carry run of the same clip may coexist
            name = shim.sanitize(f"{source.stem}_pole")
            run_dir = RUNS_DIR / name
            run_dir.mkdir(parents=True, exist_ok=True)
            motion_dir = shim.shim_motion_npz(source, run_dir)
            npz = motion_dir / f"{name}_00.npz"

            scene_params = {k: w.value
                            for k, w in self.pole_scene_widgets.items()}
            task_obj = studio_tasks.load("pole")
            try:
                task_dir, line = task_obj.reconstruct(
                    npz, run_dir / "outputs", f"{name}_00",
                    {"scene_params": scene_params})
            except Exception as e:
                self.md_pole_recon.content = f"**reconstruction crashed**: {e}"
                self.task_dir = None
                return
            if task_dir is None:
                self.md_pole_recon.content = (
                    f"**failed/skipped**\n```\n{line}\n```")
                self.task_dir = None
                return
            self.run_dir, self.task_dir = run_dir, task_dir
            manifest.update(run_dir, {
                "name": name, "source": str(source),
                "task_type": "pole", "verdict": "recon",
                "panel_scene_params": scene_params})
            self._load_reference()
            self.md_pole_recon.content = f"```\n{line.split(' -> ')[0]}\n```"
            self.md_pole_solve.content = "*ready to solve*"
        finally:
            self._set_busy(False)

    # ----------------------------------------------------------- solve --
    def do_solve(self):
        self._solve_mjwp(self.solve_widgets, self.solve_progress,
                         self.md_solve)

    def do_solve_pick(self):
        self._solve_mjwp(self.pick_solve_widgets, self.pick_progress,
                         self.md_pick_solve)

    def do_solve_pole(self):
        self._solve_mjwp(self.pole_solve_widgets, self.pole_progress,
                         self.md_pole_solve)

    def _solve_mjwp(self, widgets, progress, md):
        """The SBMPC solve + evaluation, shared by the box_carry and
        ground_pick modes (same solver, different GUI handles)."""
        if self.task_dir is None:
            return
        self._set_busy(True)
        task = self.task_dir.name
        params = {k: w.value for k, w in widgets.items()}
        logs = self.run_dir / "logs"
        logs.mkdir(exist_ok=True)
        log_path = logs / f"solve_{task}.log"
        t0 = time.time()
        md.content = "solving ..."
        # animated until the first receding-horizon step reports; warp
        # kernel compilation runs first and emits no progress
        progress.value = 0.0
        progress.animated = True
        progress.visible = True
        try:
            proc = subprocess.Popen(
                solve.command(CFG, task, self.run_dir / "outputs", params),
                # unbuffered child, so progress reaches the pipe per line
                env={**os.environ, **solve.env(), "PYTHONUNBUFFERED": "1"},
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            final = ""
            prog_re = re.compile(r"sim_steps: (\d+)/(\d+)")
            with open(log_path, "w") as log:
                for i, line in enumerate(proc.stdout):
                    log.write(line)
                    if "Final object" in line:
                        final = line.strip().split("- ")[-1]
                    m = prog_re.search(line)
                    if m:
                        cur, tot = int(m.group(1)), int(m.group(2))
                        progress.animated = False
                        progress.value = 100.0 * cur / max(tot, 1)
                        md.content = (
                            f"solving ... {time.time() - t0:.0f}s "
                            f"({cur}/{tot} sim steps)")
                    elif i % 25 == 0:
                        md.content = f"solving ... {time.time() - t0:.0f}s"
            if proc.wait() != 0:
                md.content = f"**solve failed** — see {log_path}"
                return
            self._load_result()
            metrics = solve.evaluate.evaluate_task(self.task_dir)
            row = solve.evaluate.format_row(task, metrics)
            manifest.update(self.run_dir, {
                "solve_params": params,
                "verdict": solve.evaluate.verdict([(task, metrics)]),
                "eval_rows": [row]})
            md.content = (f"**{time.time() - t0:.0f}s** — {final}\n"
                          f"```\n{row}\n```")
        finally:
            progress.visible = False
            self._set_busy(False)

    def _load_result(self):
        arr = np.load(self.task_dir / "0/trajectory_mjwp.npz")["qpos"]
        res_qpos = arr.reshape(-1, arr.shape[-1])  # (ticks*steps, nq)
        if self.res_base is None:
            self.res_base, self.res_vis, _ = self._add_robot("res_robot")
        if self.res_box is not None:
            self.res_box.remove()
        self.res_box = self.server.scene.add_mesh_simple(
            "/res_box", self._box_mesh.vertices, self._box_mesh.faces,
            color=(60, 180, 60), position=res_qpos[0, -7:-4],
            wxyz=res_qpos[0, -4:])
        for h in self.hand_handles:
            h.remove()
        hand_handles = []
        hand_names, hand_meshes, hand_poses = hand_collision_geoms(
            self.task_dir / "scene.xml", res_qpos)
        for k, name in enumerate(hand_names):
            hand_handles.append(self.server.scene.add_mesh_simple(
                f"/hand_{name}", hand_meshes[k].vertices, hand_meshes[k].faces,
                color=(230, 180, 40), opacity=0.6,
                position=hand_poses[0, k, :3], wxyz=hand_poses[0, k, 3:]))
        self.hand_handles, self.hand_poses = hand_handles, hand_poses
        self.res_has_box = True
        self.res_qpos = res_qpos  # publish LAST: handles are complete now
        self._apply_visibility()

    # --------------------------------------- obstacle-task reconstruct --
    def do_reconstruct_obs(self, tname: str):
        source = self.sources.get(self.gui_source.value)
        if source is None:
            return
        h = self.obs[tname]
        self._set_busy(True)
        h["md_recon"].content = "**building the scene ...**"
        try:
            # own run name: a box_carry run of the same clip may coexist
            name = shim.sanitize(f"{source.stem}_{OBSTACLE_TASKS[tname]['suffix']}")
            run_dir = RUNS_DIR / name
            run_dir.mkdir(parents=True, exist_ok=True)
            motion_dir = shim.shim_motion_npz(source, run_dir)
            npz = motion_dir / f"{name}_00.npz"

            scene_params = {k: w.value
                            for k, w in h["scene_widgets"].items()}
            task_obj = studio_tasks.load(tname)
            try:
                task_dir, line = task_obj.reconstruct(
                    npz, run_dir / "outputs", f"{name}_00",
                    {"scene_params": scene_params})
            except Exception as e:
                h["md_recon"].content = f"**estimation crashed**: {e}"
                self.task_dir = None
                return
            if task_dir is None:
                h["md_recon"].content = f"**skipped**\n```\n{line}\n```"
                self.task_dir = None
                return
            self.run_dir, self.task_dir = run_dir, task_dir
            manifest.update(run_dir, {
                "name": name, "source": str(source),
                "task_type": tname, "verdict": "recon",
                "panel_scene_params": scene_params})
            self._load_reference_obs(tname)
            h["md_recon"].content = f"```\n{line.split(' -> ')[0]}\n```"
            h["md_solve"].content = "*ready to solve*"
        finally:
            self._set_busy(False)

    def _load_reference_obs(self, tname: str):
        """Reference playback + the estimated obstacle boxes."""
        self._clear_ghost()
        info = json.loads((self.task_dir / "task_info.json").read_text())
        ref_qpos = np.load(
            self.task_dir / "0/trajectory_kinematic.npz")["qpos"]  # (T, nq)
        # unpublish first: tick() runs in the main thread and must not
        # touch handles while they are being rebuilt
        self.ref_qpos = None
        self.res_qpos = None
        self.hand_poses = None
        for h in (self.scene_handles + self.hand_handles
                  + self.obj_col_handles):
            h.remove()
        self.scene_handles, self.hand_handles = [], []
        self.obj_col_handles, self.obj_col_offsets = [], []
        if self.res_base is not None:
            self.res_base.visible = False
        for attr in ("res_box", "ref_box"):
            handle = getattr(self, attr)
            if handle is not None:
                handle.remove()
                setattr(self, attr, None)

        for i, prim in enumerate(info["primitives"]):
            _typ, pos, prm = prim
            color = OBSTACLE_COLORS[min(i, 1)]
            mesh = trimesh.creation.box(
                (2 * prm[0], 2 * prm[1], 2 * prm[2]))
            half = prm[3] / 2
            self.scene_handles.append(self.server.scene.add_mesh_simple(
                f"/obstacle_{i}", mesh.vertices, mesh.faces, color=color,
                opacity=0.9, position=np.asarray(pos, dtype=np.float64),
                wxyz=(np.cos(half), 0.0, 0.0, np.sin(half))))
        self._reset_frame_slider(len(ref_qpos))
        self._load_q_addr()
        self.ref_has_box = False  # robot-only model layout
        self.ref_qpos = ref_qpos  # publish LAST: handles are complete now
        self._apply_visibility()

    # ------------------------------------------------ obstacle-task solve --
    def do_solve_obs(self, tname: str):
        if self.task_dir is None:
            return
        h = self.obs[tname]
        self._set_busy(True)
        task = self.task_dir.name
        task_obj = studio_tasks.load(tname)
        params = {k: w.value for k, w in h["solve_widgets"].items()}
        logs = self.run_dir / "logs"
        logs.mkdir(exist_ok=True)
        log_path = logs / f"solve_{task}.log"
        t0 = time.time()
        h["md_solve"].content = "solving ..."
        # animated until the first receding-horizon step reports; warp
        # kernel compilation runs first and emits no progress
        h["progress"].value = 0.0
        h["progress"].animated = True
        h["progress"].visible = True
        try:
            proc = subprocess.Popen(
                task_obj.solve_command(CFG, task, self.run_dir / "outputs",
                                       params),
                env={**os.environ, **solve.env(), "PYTHONUNBUFFERED": "1"},
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            final = ""
            prog_re = re.compile(r"sim_steps: (\d+)/(\d+)")
            with open(log_path, "w") as log:
                for i, line in enumerate(proc.stdout):
                    log.write(line)
                    if line.startswith("Final"):
                        final = line.strip().split(" -> ")[0]
                    m = prog_re.search(line)
                    if m:
                        cur, tot = int(m.group(1)), int(m.group(2))
                        h["progress"].animated = False
                        h["progress"].value = 100.0 * cur / max(tot, 1)
                        h["md_solve"].content = (
                            f"solving ... {time.time() - t0:.0f}s "
                            f"({cur}/{tot} sim steps)")
                    elif i % 25 == 0:
                        h["md_solve"].content = (
                            f"solving ... {time.time() - t0:.0f}s")
            if proc.wait() != 0:
                h["md_solve"].content = f"**solve failed** — see {log_path}"
                return
            self._load_result_obs()
            rows = task_obj.evaluate([self.task_dir])
            row = task_obj.format_row(*rows[0])
            manifest.update(self.run_dir, {
                "solve_params": params,
                "verdict": task_obj.verdict(rows),
                "eval_rows": [row]})
            h["md_solve"].content = (
                f"**{time.time() - t0:.0f}s** — {final}\n```\n{row}\n```")
        finally:
            h["progress"].visible = False
            self._set_busy(False)

    def _load_result_obs(self):
        res_qpos = np.load(
            self.task_dir / "0/trajectory_aug.npz")["aug_qpos"]  # (S, nq)
        if self.res_base is None:
            self.res_base, self.res_vis, _ = self._add_robot("res_robot")
        self.hand_poses = None
        self.res_has_box = False
        self.res_qpos = res_qpos  # publish LAST
        self._apply_visibility()

    # ------------------------------------------------------ open a run --
    def open_run(self, name: str):
        """Load an already-solved run: reference, scene, and result if the
        solve has been done. This is what `studio view <name>` opens; the
        run's recorded task picks the panel mode."""
        run_dir = RUNS_DIR / name
        trial_dirs = task_dirs(run_dir)
        if not trial_dirs:
            self.md_recon.content = f"**no built trial under** `{run_dir}`"
            self.md_pick_recon.content = self.md_recon.content
            for h in self.obs.values():
                h["md_recon"].content = self.md_recon.content
            return
        task_type = manifest.read(run_dir).get("task_type") or "box_carry"
        self._set_busy(True)
        try:
            self._suppress_task_cb = True
            self.gui_task.value = task_type
            self._suppress_task_cb = False
            self._apply_task_mode()
            self.run_dir, self.task_dir = run_dir, trial_dirs[0]

            if task_type in self.obs:
                h = self.obs[task_type]
                task_obj = studio_tasks.load(task_type)
                self._load_reference_obs(task_type)
                rows = task_obj.evaluate([self.task_dir])
                if rows[0][1] is None:
                    h["md_solve"].content = "*not solved yet — ready to solve*"
                else:
                    self._load_result_obs()
                    h["md_solve"].content = (
                        f"```\n{task_obj.format_row(*rows[0])}\n```")
                h["md_recon"].content = (
                    f"*opened run* `{name}` ({trial_dirs[0].name})")
                return

            # box_carry, ground_pick and pole share the load/solve path
            md_recon, md_solve = {
                "ground_pick": (self.md_pick_recon, self.md_pick_solve),
                "pole": (self.md_pole_recon, self.md_pole_solve),
            }.get(task_type, (self.md_recon, self.md_solve))
            self._load_reference()
            metrics = solve.evaluate.evaluate_task(self.task_dir)
            if metrics is None:
                md_solve.content = "*not solved yet — ready to solve*"
            else:
                self._load_result()
                md_solve.content = (
                    f"```\n"
                    f"{solve.evaluate.format_row(trial_dirs[0].name, metrics)}"
                    f"\n```")
            md_recon.content = f"*opened run* `{name}` ({trial_dirs[0].name})"
        finally:
            self._set_busy(False)

    # -------------------------------------------------------- playback --
    def tick(self):
        # snapshot: worker threads swap these while we run
        q, ref_box = self.ref_qpos, self.ref_box
        if q is None:
            return
        T = len(q)
        if self.gui_play.value:
            self.gui_frame.value = (int(self.gui_frame.value) + 1) % T
        t = min(int(self.gui_frame.value), T - 1)
        try:
            self.ref_base.position = q[t, :3]
            self.ref_base.wxyz = q[t, 3:7]
            self.ref_vis.update_cfg(self._cfg(q[t]))
            if self.ref_has_box and ref_box is not None:  # reconstructed ref
                ref_box.position = q[t, -7:-4]
                ref_box.wxyz = q[t, -4:]
            g, gb = self.ghost, self.ghost_box
            if q.shape[1] == layout.BODY_NQ and g is not None and gb is not None:
                # build_object_trajectory's rule: rest at the pick pose,
                # ride the hands in the window, freeze at release
                s = min(max(t, g["pick"]), g["rel"])
                gb.position = self._mid[s]
                half = self._yaw[s] / 2
                gb.wxyz = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
            r, res_box = self.res_qpos, self.res_box
            hand_poses, hand_handles = self.hand_poses, self.hand_handles
            if r is not None and self.res_base is not None:
                s = min(t * 2, len(r) - 1)  # result 60 Hz vs ref 30
                self.res_base.position = r[s, :3]
                self.res_base.wxyz = r[s, 3:7]
                self.res_vis.update_cfg(self._cfg(r[s]))
                if self.res_has_box and res_box is not None:  # box result
                    res_box.position = r[s, -7:-4]
                    res_box.wxyz = r[s, -4:]
                if hand_poses is not None and len(hand_poses) == len(r):
                    for k, h in enumerate(hand_handles):
                        h.position = hand_poses[s, k, :3]
                        h.wxyz = hand_poses[s, k, 3:]
            # follow the result's object pose when one is shown
            oc_handles, oc_offsets = self.obj_col_handles, self.obj_col_offsets
            if oc_handles and self.gui_show_objcol.value:
                if (r is not None and self.res_has_box
                        and self.gui_show_res.value):
                    qo = r[min(t * 2, len(r) - 1), -7:]
                elif self.ref_has_box:
                    qo = q[t, -7:]
                else:
                    qo = None
                if qo is not None:
                    R = np.zeros(9)
                    mujoco.mju_quat2Mat(R, qo[3:])
                    R = R.reshape(3, 3)
                    quat = np.zeros(4)
                    for k, h in enumerate(oc_handles):
                        p_off, q_off = oc_offsets[k]
                        h.position = qo[:3] + R @ p_off
                        mujoco.mju_mulQuat(quat, qo[3:], q_off)
                        h.wxyz = quat.copy()
        except RuntimeError:
            # a worker replaced a snapshotted handle mid-frame; the next
            # tick renders the fresh state
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--run", default=None,
                    help="open an existing run instead of a raw clip")
    args = ap.parse_args()
    server = viser.ViserServer(port=args.port)
    panel = Panel(server)
    if args.run:
        # wait out the constructor's initial preview, or _spawn silently
        # drops the open while the busy flag is held
        def _open():
            while panel.busy:
                time.sleep(0.1)
            panel.open_run(args.run)
        threading.Thread(target=_open, daemon=True).start()
    print(f"panel at http://localhost:{args.port}", flush=True)
    import traceback
    while True:
        try:
            panel.tick()
        except Exception:  # a transient race must never kill the server
            traceback.print_exc()
        time.sleep(1.0 / (30.0 * panel.gui_speed.value))


if __name__ == "__main__":
    main()
