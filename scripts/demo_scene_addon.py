"""Kimodo viser demo + scene-reconstruction add-on.

Wraps the stock demo, leaving kimodo pristine: constructs `Demo`,
attaches an extra GUI folder to the same viser server, then hands control
to `demo.run()`. The Reconstruct button saves the committed sample into
raw_motion/, shells out to `studio recon` (CPU only, so no VRAM cost next
to the diffusion model), and overlays the reconstructed scene on the
playing motion. The task dropdown lists every registered task
(studio.tasks, MuJoCo-free), each with the controls its scene defaults
and choices describe. Solving stays in the separate solve panel
(`studio panel`).

The demo world is y-up and the reconstruction MuJoCo z-up, so the whole
overlay lives under one parent frame rotated by the inverse permutation
and all child poses stay in z-up scene coordinates.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import viser

STUDIO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDIO_ROOT / "src"))

from studio import viz
from studio import tasks as studio_tasks
from studio.config import (RAW_MOTION_DIR, RUNS_DIR, SCENE_DEFAULTS,
                           load_config, task_dirs)
from studio.shim import sanitize

from kimodo.demo.app import Demo
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.model.registry import resolve_model_name

CFG = load_config()
# kim = R @ mj with mj = [kim_z, kim_x, kim_y]: -120deg about (1,1,1)/sqrt(3)
ZUP_TO_YUP_WXYZ = (0.5, -0.5, -0.5, -0.5)


def studio_bin() -> str:
    """Locate the `studio` CLI. This add-on runs in the kimodo venv, which
    has no MuJoCo, so reconstruction goes back out through studio's venv."""
    found = os.environ.get("STUDIO_BIN") or shutil.which("studio")
    if found:
        return found
    local = STUDIO_ROOT / ".venv/bin/studio"
    if local.exists():
        return str(local)
    raise RuntimeError("cannot find the `studio` CLI — run `uv sync`, or "
                       "point STUDIO_BIN at it")


class SceneReconAddon:
    def __init__(self, demo: Demo):
        self.demo = demo
        self.server = demo.server
        self.busy = False
        self.state = None       # {"box": (T,7), "sid": client_id} — publish LAST
        self.box_handle = None
        self.static_handles = []
        self.root = self.server.scene.add_frame(
            "/scene_recon", show_axes=False, wxyz=ZUP_TO_YUP_WXYZ)
        self._build_gui()
        threading.Thread(target=self._anim_loop, daemon=True).start()

    def _build_gui(self):
        gui = self.server.gui
        with gui.add_folder("Scene recon"):
            self.gui_name = gui.add_text("run name", initial_value="clip")
            self.gui_task = gui.add_dropdown(
                "task", tuple(studio_tasks.names()),
                initial_value=studio_tasks.DEFAULT)
            self.folder_box = gui.add_folder("box_carry params")
            with self.folder_box:
                self.scene_widgets = viz.scene_param_widgets(
                    gui, SCENE_DEFAULTS,
                    studio_tasks.load(studio_tasks.DEFAULT).choices)
                # contact window (the period the robot holds the object);
                # -1 = full clip. "auto window" hands the choice to the grasp
                # detector instead ("allow held start" lets it accept a clip
                # that begins mid-hold); the reconstruct status line reports
                # the window actually used, and after an auto run the detected
                # frames land in the fields — untick auto to correct from
                # there and re-click.
                self.gui_auto = gui.add_checkbox("auto window", False)
                self.gui_held = gui.add_checkbox("allow held start", False,
                                                 disabled=True)
                self.gui_cstart = gui.add_number("contact start", -1, min=-1,
                                                 step=1)
                self.gui_cend = gui.add_number("contact end", -1, min=-1,
                                               step=1)
            # every other task: the controls its scene defaults describe;
            # the detectors find the window and hands on their own
            self.task_folders = {}
            for tname in studio_tasks.names():
                if tname == studio_tasks.DEFAULT:
                    continue
                task = studio_tasks.load(tname)
                folder = gui.add_folder(f"{tname} params", visible=False)
                with folder:
                    widgets = viz.scene_param_widgets(
                        gui, task.scene_defaults, task.choices, step=0.01)
                self.task_folders[tname] = (folder, widgets)
            self.gui_show = gui.add_checkbox("show scene", True)
            self.btn = gui.add_button("Reconstruct scene", color="teal")
            self.md = gui.add_markdown(
                "*generate + click a sample, then reconstruct*")

        @self.gui_show.on_update
        def _(_):
            self.root.visible = self.gui_show.value

        @self.gui_task.on_update
        def _(_):
            active = self.gui_task.value
            self.folder_box.visible = active == studio_tasks.DEFAULT
            for tname, (folder, _w) in self.task_folders.items():
                folder.visible = tname == active

        @self.gui_auto.on_update
        def _(_):
            auto = self.gui_auto.value
            self.gui_cstart.disabled = auto
            self.gui_cend.disabled = auto
            self.gui_held.disabled = not auto

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

            if self.gui_task.value in self.task_folders:
                # the task's detector needs no window flags; its params
                # travel as --set (the box flags are box_carry-only)
                flags = ["--task", self.gui_task.value]
                for k, w in self.task_folders[self.gui_task.value][1].items():
                    flags += ["--set", f"{k}={w.value}"]
            else:
                flags = []
                for k, w in self.scene_widgets.items():
                    flags += [f"--{k.replace('_', '-')}", str(w.value)]
                if self.gui_auto.value:
                    # no window flags: recon's grasp detector picks the
                    # window
                    if self.gui_held.value:
                        flags.append("--allow-held-start")
                else:
                    # explicit window: -1 defaults to the full clip (start
                    # at 0 implies a held start — box spawns in the hands)
                    T = int(motion.joints_pos.shape[0])
                    cs = int(self.gui_cstart.value)
                    ce = int(self.gui_cend.value)
                    flags += ["--pick-frame", str(cs if cs >= 0 else 0),
                              "--release-frame", str(ce if ce >= 0 else T - 1)]
            proc = subprocess.run(
                [studio_bin(), "recon", str(npz_path), "--name", name,
                 *flags],
                capture_output=True, text=True)
            pick_line = next(
                (ln for ln in proc.stdout.splitlines()
                 if "pick f" in ln or "hold f" in ln), "")
            tasks = task_dirs(RUNS_DIR / name, prefix=name)
            if proc.returncode != 0 or not tasks:
                tail = (pick_line or proc.stdout.strip().splitlines()[-1:]
                        or proc.stderr.strip().splitlines()[-1:] or ["?"])
                tail = tail if isinstance(tail, str) else "\n".join(tail)
                self.md.content = f"**failed/skipped**\n```\n{tail}\n```"
                return
            self._load_overlay(tasks[0], client_id)
            if (self.gui_task.value == studio_tasks.DEFAULT
                    and self.gui_auto.value):
                # surface the detected window in the fields, so unticking
                # auto lets the user correct from it instead of from -1
                m = re.search(r"pick f(\d+) release f(\d+)", pick_line)
                if m:
                    self.gui_cstart.value = int(m.group(1))
                    self.gui_cend.value = int(m.group(2))
            self.md.content = (
                f"```\n{pick_line.split(' -> ')[0]}\n```\n"
                f"saved `runs/{name}` — solve it in `studio panel`")
        except Exception as e:  # surface instead of dying silently
            self.md.content = f"**error**: {e}"
        finally:
            self.busy = False
            self.btn.disabled = False

    def _load_overlay(self, task_dir: Path, client_id: int):
        info = viz.read_info(task_dir)
        ref_qpos = np.load(task_dir / "0/trajectory_kinematic.npz")["qpos"]
        # the object free joint is always last in qpos, whatever the robot
        # layout (see studio.recon.layout — not importable here: no mujoco)
        box_traj = ref_qpos[:, -7:].copy()

        self.state = None  # unpublish: anim thread must not touch old handles
        for h in [self.box_handle, *self.static_handles]:
            if h is not None:
                h.remove()
        self.static_handles, self.box_handle = [], None

        self.static_handles = viz.add_scene(
            self.server, task_dir, prefix="/scene_recon", opacity=0.8)
        mesh = viz.object_mesh(info)
        self.box_handle = self.server.scene.add_mesh_simple(
            "/scene_recon/box", mesh.vertices, mesh.faces,
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
    print("scene-recon add-on attached (folder 'Scene recon')",
          flush=True)
    demo.run()


if __name__ == "__main__":
    main()
