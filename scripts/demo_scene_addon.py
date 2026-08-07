"""Kimodo viser demo + scene-reconstruction add-on.

Wraps the stock demo (kimodo stays pristine): constructs `Demo`, attaches
an extra GUI folder to the same viser server, then hands control to
`demo.run()`. The Reconstruct button saves the committed sample as a
kimodo NPZ into raw_motion/, shells out to `studio recon` (build_trial in
mppi_locoma's venv — CPU only, no VRAM cost next to the diffusion model),
and overlays the reconstructed scene (table, terrain, animated box) on the
playing motion. Solving stays in the separate solve panel (`studio panel`).

The demo world is y-up (kimodo convention); the reconstruction is MuJoCo
z-up. The loader maps mj = [kim_z, kim_x, kim_y], so the whole overlay
lives under one parent frame rotated by the inverse permutation and all
child poses stay in z-up scene coordinates.
"""

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

STUDIO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDIO_ROOT / "src"))

from studio.config import RUNS_DIR, TASK_SUBTREE, load_config
from studio.shim import sanitize

from kimodo.demo.app import Demo
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.model.registry import resolve_model_name

CFG = load_config()
RAW_MOTION_DIR = STUDIO_ROOT / "raw_motion"
STUDIO_BIN = STUDIO_ROOT / ".venv/bin/studio"
# kim = R @ mj with mj = [kim_z, kim_x, kim_y]: -120deg about (1,1,1)/sqrt(3)
ZUP_TO_YUP_WXYZ = (0.5, -0.5, -0.5, -0.5)


def scene_defaults() -> dict:
    code = ("import sys, json; sys.path.insert(0, sys.argv[1]); "
            "from repo_config import SCENE_DEFAULTS; "
            "print(json.dumps(SCENE_DEFAULTS))")
    out = subprocess.run(
        [str(CFG.mppi_python), "-c", code, str(CFG.mppi_locoma / "src")],
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


class SceneReconAddon:
    def __init__(self, demo: Demo):
        self.demo = demo
        self.server = demo.server
        self.busy = False
        self.state = None       # {"box": (T,7), "sid": client_id} — publish LAST
        self.box_handle = None
        self.static_handles = []
        self.root = self.server.scene.add_frame(
            "/mppi_recon", show_axes=False, wxyz=ZUP_TO_YUP_WXYZ)
        self._build_gui()
        threading.Thread(target=self._anim_loop, daemon=True).start()

    def _build_gui(self):
        gui = self.server.gui
        with gui.add_folder("Scene recon (mppi)"):
            self.gui_name = gui.add_text("run name", initial_value="clip")
            self.scene_widgets = {}
            for key, val in scene_defaults().items():
                if key == "hand_geom":
                    self.scene_widgets[key] = gui.add_dropdown(
                        key, ("mesh", "capsule"), initial_value=val)
                else:
                    self.scene_widgets[key] = gui.add_number(
                        key, float(val), min=0.0, step=0.005)
            self.gui_show = gui.add_checkbox("show scene", True)
            self.btn = gui.add_button("Reconstruct scene", color="teal")
            self.md = gui.add_markdown(
                "*generate + click a sample, then reconstruct*")

        @self.gui_show.on_update
        def _(_):
            self.root.visible = self.gui_show.value

        @self.btn.on_click
        def _(event: viser.GuiEvent):
            if self.busy or event.client is None:
                return
            threading.Thread(target=self.do_recon,
                             args=(event.client.client_id,), daemon=True).start()

    # ---------------------------------------------------------- recon --
    def do_recon(self, client_id: int):
        self.busy = True
        self.btn.disabled = True
        try:
            session = self.demo.client_sessions.get(client_id)
            if session is None or not session.motions:
                self.md.content = "**no motion** — generate one first"
                return
            motion = list(session.motions.values())[0]
            if motion.foot_contacts is None:
                self.md.content = ("**no foot contacts on this motion** — "
                                   "reconstruction needs them")
                return
            name = sanitize(self.gui_name.value)
            self.gui_name.value = name
            npz_path = RAW_MOTION_DIR / f"{name}.npz"
            RAW_MOTION_DIR.mkdir(exist_ok=True)
            self.md.content = "**reconstructing ...** (~10-20 s)"
            save_kimodo_npz(str(npz_path), {
                "posed_joints": motion.joints_pos.detach().cpu().numpy(),
                "global_rot_mats": motion.joints_rot.detach().cpu().numpy(),
                "local_rot_mats": motion.joints_local_rot.detach().cpu().numpy(),
                "root_positions": motion.joints_pos.detach().cpu().numpy()
                [:, motion.skeleton.root_idx, :],
                "foot_contacts": motion.foot_contacts.detach().cpu().numpy(),
            })

            flags = []
            for k, w in self.scene_widgets.items():
                flags += [f"--{k.replace('_', '-')}", str(w.value)]
            proc = subprocess.run(
                [str(STUDIO_BIN), "recon", str(npz_path), "--name", name,
                 *flags],
                capture_output=True, text=True)
            pick_line = next(
                (ln for ln in proc.stdout.splitlines() if "pick f" in ln), "")
            run_dir = RUNS_DIR / name
            task_root = run_dir / "outputs" / TASK_SUBTREE
            tasks = (sorted(d for d in task_root.iterdir() if d.is_dir()
                            and d.name.startswith(name))
                     if task_root.is_dir() else [])
            if proc.returncode != 0 or not tasks:
                tail = (pick_line or proc.stdout.strip().splitlines()[-1:]
                        or proc.stderr.strip().splitlines()[-1:] or ["?"])
                tail = tail if isinstance(tail, str) else "\n".join(tail)
                self.md.content = f"**failed/skipped**\n```\n{tail}\n```"
                return
            self._load_overlay(tasks[0], client_id)
            self.md.content = (
                f"```\n{pick_line.split(' -> ')[0]}\n```\n"
                f"saved `runs/{name}` — solve it in `studio panel`")
        except Exception as e:  # surface instead of dying silently
            self.md.content = f"**error**: {e}"
        finally:
            self.busy = False
            self.btn.disabled = False

    def _load_overlay(self, task_dir: Path, client_id: int):
        info = json.loads((task_dir / "task_info.json").read_text())
        ref_qpos = np.load(task_dir / "0/trajectory_kinematic.npz")["qpos"]
        box_traj = ref_qpos[:, 36:43].copy()

        self.state = None  # unpublish: anim thread must not touch old handles
        for h in [self.box_handle, *self.static_handles]:
            if h is not None:
                h.remove()
        self.static_handles, self.box_handle = [], None

        add = self.server.scene.add_mesh_simple
        if info.get("has_table"):
            m = re.search(r'<geom name="table"[^>]*size="([^"]+)"[^>]*'
                          r'pos="([^"]+)"(?:[^>]*quat="([^"]+)")?',
                          (task_dir / "scene.xml").read_text())
            if m:
                mesh = trimesh.creation.box(
                    np.fromstring(m.group(1), sep=" ") * 2)
                self.static_handles.append(add(
                    "/mppi_recon/table", mesh.vertices, mesh.faces,
                    color=(140, 100, 60), opacity=0.8,
                    position=np.fromstring(m.group(2), sep=" "),
                    wxyz=(np.fromstring(m.group(3), sep=" ") if m.group(3)
                          else np.array([1.0, 0, 0, 0]))))
        for i, g in enumerate(info.get("terrain_geoms", [])):
            mesh = trimesh.creation.box(np.array(g["half"]) * 2)
            yaw = g.get("yaw", 0.0)
            self.static_handles.append(add(
                f"/mppi_recon/terrain_{i}", mesh.vertices, mesh.faces,
                color=(150, 140, 120), opacity=0.8,
                position=np.array(g["center"]),
                wxyz=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])))
        box_mesh = trimesh.creation.box(info["box_size"])
        self.box_handle = add(
            "/mppi_recon/box", box_mesh.vertices, box_mesh.faces,
            color=(200, 60, 60), opacity=0.6,
            position=box_traj[0, :3], wxyz=box_traj[0, 3:])
        self.state = {"box": box_traj, "sid": client_id}  # publish LAST

    # ------------------------------------------------------- animation --
    def _anim_loop(self):
        while True:
            st, box = self.state, self.box_handle
            if st is not None and box is not None:
                session = self.demo.client_sessions.get(st["sid"])
                if session is not None:
                    t = min(session.frame_idx, len(st["box"]) - 1)
                    try:
                        box.position = st["box"][t, :3]
                        box.wxyz = st["box"][t, 3:]
                    except RuntimeError:
                        pass  # handle replaced mid-frame; next tick recovers
            time.sleep(1.0 / 30.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Kimodo demo + scene recon")
    ap.add_argument("--model", type=str, default=CFG.model)
    args = ap.parse_args()
    demo = Demo(default_model_name=resolve_model_name(args.model, "Kimodo"))
    SceneReconAddon(demo)
    print("scene-recon add-on attached (folder 'Scene recon (mppi)')",
          flush=True)
    demo.run()


if __name__ == "__main__":
    main()
