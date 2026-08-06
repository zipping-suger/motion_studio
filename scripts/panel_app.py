"""Interactive tuning panel: reconstruct + SBMPC-solve from viser buttons.

Pick a motion (raw_motion/*.npz or a saved demo example), click
"Reconstruct scene" (mppi_locoma build_trial with the scene params set in
the GUI), inspect the reconstruction, tune SBMPC hyperparameters, click
"Solve SBMPC" (run_mjwp), and watch reference (transparent) vs solution
(solid) playback. Every action works inside an ordinary studio run dir
(runs/<name>/), so list/view/promote all apply.

Runs in mppi_locoma's venv (viser + mujoco): launch with `studio panel`.
Visualization conventions mirror mppi_locoma/scripts/view_trial.py.
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
import yaml

STUDIO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDIO_ROOT / "src"))

from studio import manifest, shim
from studio.config import RUNS_DIR, TASK_SUBTREE, load_config

CFG = load_config()
sys.path.insert(0, str(CFG.mppi_locoma / "src"))

import mujoco
import spider
import viser
from viser.extras import ViserUrdf
import yourdfpy

from kimodo_loader import DOF_NAMES
from repo_config import G1_URDF, RETARGET_PARAMS, SCENE_DEFAULTS

MESHES_DIR = str(Path(spider.ROOT) / "assets/robots/unitree_g1/meshes")
INT_KEYS = {"num_samples", "max_num_iterations"}


def hand_collision_geoms(scene_xml, qpos_traj):
    """World poses of the lh/rh collision geoms via FK (view_trial.py)."""
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            for g in ("lh", "rh")]
    meshes = []
    for gid, side in zip(gids, ("left", "right")):
        if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            hand = trimesh.load(f"{MESHES_DIR}/{side}_rubber_hand.STL")
            hull = trimesh.convex.convex_hull(hand)
            mid = m.geom_dataid[gid]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.mesh_quat[mid])
            hull.vertices = (hull.vertices - m.mesh_pos[mid]) @ R.reshape(3, 3)
            meshes.append(hull)
        else:
            radius, half_len = m.geom_size[gid][0], m.geom_size[gid][1]
            meshes.append(trimesh.creation.capsule(radius=radius,
                                                   height=2 * half_len))
    poses = np.zeros((len(qpos_traj), 2, 7))
    quat = np.zeros(4)
    for t, qp in enumerate(qpos_traj):
        d.qpos[:] = qp
        mujoco.mj_forward(m, d)
        for k, gid in enumerate(gids):
            poses[t, k, :3] = d.geom_xpos[gid]
            mujoco.mju_mat2Quat(quat, d.geom_xmat[gid].ravel())
            poses[t, k, 3:] = quat
    return meshes, poses


def list_sources():
    out = {}
    raw = STUDIO_ROOT / "raw_motion"
    if raw.is_dir():
        for f in sorted(raw.glob("*.npz")):
            out[f"npz: {f.stem}"] = f
    if CFG.examples_dir.is_dir():
        for d in sorted(CFG.examples_dir.iterdir()):
            if (d / "motion.npz").exists():
                out[f"example: {d.name}"] = d
    return out


class Panel:
    def __init__(self, server: viser.ViserServer):
        self.server = server
        self.run_dir = None
        self.task_dir = None
        self.ref_qpos = None
        self.res_qpos = None
        self.hand_poses = None
        self.hand_handles = []
        self.scene_handles = []
        self.busy = False

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
        order = [DOF_NAMES.index(n) for n in urdf_model.actuated_joint_names]
        return base, vis, order

    # ------------------------------------------------------------- GUI --
    def _build_gui(self):
        gui = self.server.gui
        with gui.add_folder("Motion"):
            self.sources = list_sources()
            options = list(self.sources) or ["<none found>"]
            self.gui_source = gui.add_dropdown("source", options,
                                               initial_value=options[0])
            refresh = gui.add_button("refresh sources")

            @refresh.on_click
            def _(_):
                self.sources = list_sources()
                opts = list(self.sources) or ["<none found>"]
                self.gui_source.options = opts

        with gui.add_folder("Scene reconstruction"):
            self.scene_widgets = {}
            for key, val in SCENE_DEFAULTS.items():
                if key == "hand_geom":
                    self.scene_widgets[key] = gui.add_dropdown(
                        key, ("mesh", "capsule"), initial_value=val)
                else:
                    v = float(val)
                    self.scene_widgets[key] = gui.add_number(
                        key, v, min=0.0, step=0.005)
            self.btn_recon = gui.add_button("1. Reconstruct scene")
            self.md_recon = gui.add_markdown("*no reconstruction yet*")

        with gui.add_folder("SBMPC solve"):
            self.solve_widgets = {}
            for key, val in RETARGET_PARAMS.items():
                if key in INT_KEYS:
                    self.solve_widgets[key] = gui.add_number(
                        key, int(val), min=1, step=max(1, int(val) // 16))
                else:
                    self.solve_widgets[key] = gui.add_number(
                        key, float(val), min=0.0, step=0.05)
            self.btn_solve = gui.add_button("2. Solve SBMPC")
            self.btn_solve.disabled = True
            self.md_solve = gui.add_markdown("*reconstruct first*")

        with gui.add_folder("Playback"):
            self.gui_frame = gui.add_slider("progress", 0.0, 1.0, 0.001, 0.0)
            self.gui_play = gui.add_checkbox("play", True)
            self.gui_speed = gui.add_slider("speed", 0.1, 2.0, 0.1, 1.0)

        @self.btn_recon.on_click
        def _(_):
            self._spawn(self.do_reconstruct)

        @self.btn_solve.on_click
        def _(_):
            self._spawn(self.do_solve)

    def _spawn(self, fn):
        if self.busy:
            return
        threading.Thread(target=fn, daemon=True).start()

    def _set_busy(self, busy: bool):
        self.busy = busy
        self.btn_recon.disabled = busy
        self.btn_solve.disabled = busy or self.task_dir is None

    # ----------------------------------------------------- reconstruct --
    def do_reconstruct(self):
        label = self.gui_source.value
        source = self.sources.get(label)
        if source is None:
            return
        self._set_busy(True)
        self.md_recon.content = "**reconstructing ...** (takes ~10-20 s)"
        try:
            name = shim.sanitize(source.stem if source.is_file()
                                 else source.name)
            run_dir = RUNS_DIR / name
            run_dir.mkdir(parents=True, exist_ok=True)
            if source.is_file():
                motion_dir = shim.shim_motion_npz(source, run_dir)
            else:
                motion_dir = shim.shim_example(source, run_dir)

            scene_params = {k: w.value for k, w in self.scene_widgets.items()}
            flags = []
            for k, v in scene_params.items():
                flags += [f"--{k.replace('_', '-')}", str(v)]
            out_root = run_dir / "outputs"
            proc = subprocess.run(
                [str(CFG.mppi_python),
                 str(CFG.mppi_locoma / "scripts/build_trial.py"),
                 "--motion-dir", str(motion_dir), "--out-root", str(out_root),
                 *flags],
                cwd=str(CFG.mppi_locoma), capture_output=True, text=True)
            result_line = next(
                (ln for ln in proc.stdout.splitlines() if "pick f" in ln),
                proc.stdout.strip().splitlines()[-1] if proc.stdout.strip()
                else proc.stderr.strip().splitlines()[-1] if proc.stderr.strip()
                else "no output")
            task_root = out_root / TASK_SUBTREE
            tasks = (sorted(d for d in task_root.iterdir() if d.is_dir())
                     if task_root.is_dir() else [])
            if proc.returncode != 0 or not tasks or "SKIPPED" in result_line:
                self.md_recon.content = f"**failed/skipped**\n```\n{result_line}\n```"
                self.task_dir = None
                return
            self.run_dir, self.task_dir = run_dir, tasks[0]
            manifest.update(run_dir, {"name": name, "source": str(source),
                                      "panel_scene_params": scene_params})
            self._load_reference()
            self.md_recon.content = f"```\n{result_line.split(' -> ')[0]}\n```"
            self.md_solve.content = "*ready to solve*"
        finally:
            self._set_busy(False)

    def _load_reference(self):
        info = json.loads((self.task_dir / "task_info.json").read_text())
        ref_qpos = np.load(
            self.task_dir / "0/trajectory_kinematic.npz")["qpos"]
        # unpublish first: tick() runs concurrently in the main thread and
        # must not touch handles while they are being rebuilt
        self.ref_qpos = None
        self.res_qpos = None
        for h in self.scene_handles + self.hand_handles:
            h.remove()
        self.scene_handles, self.hand_handles = [], []
        if self.res_base is not None:
            self.res_base.visible = False
        if self.res_box is not None:
            self.res_box.remove()
            self.res_box = None

        add = self.server.scene.add_mesh_simple
        if info.get("has_table"):
            xml = (self.task_dir / "scene.xml").read_text()
            m = re.search(r'<geom name="table"[^>]*size="([^"]+)"[^>]*'
                          r'pos="([^"]+)"(?:[^>]*quat="([^"]+)")?', xml)
            if m:
                size = np.fromstring(m.group(1), sep=" ") * 2
                mesh = trimesh.creation.box(size)
                self.scene_handles.append(add(
                    "/scene/table", mesh.vertices, mesh.faces,
                    color=(140, 100, 60),
                    position=np.fromstring(m.group(2), sep=" "),
                    wxyz=(np.fromstring(m.group(3), sep=" ") if m.group(3)
                          else np.array([1.0, 0, 0, 0]))))
        for i, g in enumerate(info.get("terrain_geoms", [])):
            mesh = trimesh.creation.box(np.array(g["half"]) * 2)
            yaw = g.get("yaw", 0.0)
            self.scene_handles.append(add(
                f"/scene/terrain_{i}", mesh.vertices, mesh.faces,
                color=(150, 140, 120), position=np.array(g["center"]),
                wxyz=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])))

        if self.ref_box is not None:
            self.ref_box.remove()
        box_mesh = trimesh.creation.box(info["box_size"])
        self.ref_box = add("/ref_box", box_mesh.vertices, box_mesh.faces,
                           color=(200, 60, 60), opacity=0.4,
                           position=ref_qpos[0, 36:39],
                           wxyz=ref_qpos[0, 39:43])
        self._box_mesh = box_mesh
        self.ref_base.visible = True
        self.gui_frame.value = 0.0
        self.ref_qpos = ref_qpos  # publish LAST: handles are complete now

    # ----------------------------------------------------------- solve --
    def do_solve(self):
        if self.task_dir is None:
            return
        self._set_busy(True)
        task = self.task_dir.name
        params = {k: w.value for k, w in self.solve_widgets.items()}
        overrides = [f"{k}={int(v) if k in INT_KEYS else float(v)}"
                     for k, v in params.items()]
        logs = self.run_dir / "logs"
        logs.mkdir(exist_ok=True)
        log_path = logs / f"retarget_{task}.log"
        t0 = time.time()
        self.md_solve.content = "solving ..."
        try:
            proc = subprocess.Popen(
                [str(CFG.mppi_python),
                 str(CFG.mppi_locoma / "retarget/run_mjwp.py"),
                 "+override=kimodo_pick", f"task={task}",
                 f"dataset_dir={self.run_dir / 'outputs'}", *overrides,
                 "show_viewer=false", "+wait_on_finish=false",
                 "save_video=false", "+sanity_check_seconds=0.0"],
                cwd=str(CFG.mppi_locoma),
                env={**os.environ, "MUJOCO_GL": "egl"},
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            final = ""
            with open(log_path, "w") as log:
                for i, line in enumerate(proc.stdout):
                    log.write(line)
                    if "Final object" in line:
                        final = line.strip().split("- ")[-1]
                    if i % 25 == 0:
                        self.md_solve.content = (
                            f"solving ... {time.time() - t0:.0f}s")
            if proc.wait() != 0:
                self.md_solve.content = f"**solve failed** — see {log_path}"
                return
            self._load_result()
            row = self._eval(task)
            manifest.update(self.run_dir, {
                "panel_retarget_params": params,
                "verdict": "LIFT" if " YES" in row else "failed",
                "eval_rows": [row]})
            self.md_solve.content = (f"**{time.time() - t0:.0f}s** — {final}\n"
                                     f"```\n{row}\n```")
        finally:
            self._set_busy(False)

    def _load_result(self):
        res_qpos = np.load(
            self.task_dir / "0/trajectory_mjwp.npz")["qpos"].reshape(-1, 43)
        if self.res_base is None:
            self.res_base, self.res_vis, _ = self._add_robot("res_robot")
        self.res_base.visible = True
        if self.res_box is not None:
            self.res_box.remove()
        self.res_box = self.server.scene.add_mesh_simple(
            "/res_box", self._box_mesh.vertices, self._box_mesh.faces,
            color=(60, 180, 60), position=res_qpos[0, 36:39],
            wxyz=res_qpos[0, 39:43])
        for h in self.hand_handles:
            h.remove()
        hand_handles = []
        hand_meshes, hand_poses = hand_collision_geoms(
            self.task_dir / "scene.xml", res_qpos)
        for k, side in enumerate(("lh", "rh")):
            hand_handles.append(self.server.scene.add_mesh_simple(
                f"/hand_{side}", hand_meshes[k].vertices, hand_meshes[k].faces,
                color=(230, 180, 40), opacity=0.6,
                position=hand_poses[0, k, :3], wxyz=hand_poses[0, k, 3:]))
        self.hand_handles, self.hand_poses = hand_handles, hand_poses
        self.res_qpos = res_qpos  # publish LAST: handles are complete now

    def _eval(self, task):
        out = subprocess.run(
            [str(CFG.mppi_python),
             str(CFG.mppi_locoma / "scripts/eval_trials.py"),
             "--root", str(self.run_dir / "outputs" / TASK_SUBTREE),
             "--filter", task],
            cwd=str(CFG.mppi_locoma), capture_output=True, text=True).stdout
        return next((ln for ln in out.splitlines() if ln.startswith(task)),
                    out.strip())

    # -------------------------------------------------------- playback --
    def tick(self):
        # snapshot: worker threads swap these while we run
        q, ref_box = self.ref_qpos, self.ref_box
        if q is None or ref_box is None:
            return
        T = len(q)
        if self.gui_play.value:
            self.gui_frame.value = (self.gui_frame.value
                                    + 1.0 / max(T - 1, 1)) % 1.0
        t = min(int(self.gui_frame.value * (T - 1)), T - 1)
        self.ref_base.position = q[t, :3]
        self.ref_base.wxyz = q[t, 3:7]
        self.ref_vis.update_cfg(q[t, 7:36][self.order])
        ref_box.position = q[t, 36:39]
        ref_box.wxyz = q[t, 39:43]
        r, res_box = self.res_qpos, self.res_box
        hand_poses, hand_handles = self.hand_poses, self.hand_handles
        if r is not None and self.res_base is not None and res_box is not None:
            s = min(t * 2, len(r) - 1)  # result 60 Hz vs ref 30
            self.res_base.position = r[s, :3]
            self.res_base.wxyz = r[s, 3:7]
            self.res_vis.update_cfg(r[s, 7:36][self.order])
            res_box.position = r[s, 36:39]
            res_box.wxyz = r[s, 39:43]
            if hand_poses is not None and len(hand_poses) == len(r):
                for k, h in enumerate(hand_handles):
                    h.position = hand_poses[s, k, :3]
                    h.wxyz = hand_poses[s, k, 3:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8082)
    args = ap.parse_args()
    server = viser.ViserServer(port=args.port)
    panel = Panel(server)
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
