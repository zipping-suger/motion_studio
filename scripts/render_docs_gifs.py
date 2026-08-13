"""Render the three README GIFs headlessly: generation, recon, solve.

Uses the solve venv (mujoco + Pillow) and ffmpeg for palette-quantized
GIF encoding:

    MUJOCO_GL=egl .venv-solve/bin/python scripts/render_docs_gifs.py

Inputs are the checked-in sample run: the raw Kimodo clip
(raw_motion/box_carrying.npz), and the recon + solve artifacts under
runs/box_carrying/. Outputs land in docs/.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = (ROOT / "runs/box_carrying/outputs/processed/kimodo/unitree_g1"
            / "humanoid_object/box_carrying_00")

FPS_OUT = 15          # reference is 30 fps; every 2nd frame
W, H = 640, 400       # single-panel size; solve gif is two panels wide
PANEL_W = 476

# Kimodo 34-joint skeleton parents (see studio.recon.loader docstring)
PARENTS = [-1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 0, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24, 17, 26, 27, 28, 29, 30, 31, 32]

BONE_RGBA = np.array([0.20, 0.28, 0.45, 1.0], dtype=np.float32)
JOINT_RGBA = np.array([0.85, 0.35, 0.10, 1.0], dtype=np.float32)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

FLOOR_XML = """
<mujoco model="skeleton stage">
  <visual>
    <global azimuth="135" elevation="-25" offwidth="1920" offheight="1080"/>
    <quality shadowsize="8192"/>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6" specular="0 0 0"/>
    <rgba haze="0.94 0.95 0.97 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1"
             width="3072" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.9 0.9 0.9" rgb2="0.8 0.8 0.8" markrgb="0.4 0.4 0.4"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="4 4"/>
  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" type="directional"/>
    <geom name="floor" size="0 0 0.01" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""


def load_skeleton() -> np.ndarray:
    """Raw Kimodo posed joints, converted y-up -> z-up. (T, 34, 3)."""
    pj = np.load(ROOT / "raw_motion/box_carrying.npz")["posed_joints"]
    pj = pj.astype(np.float64)
    return np.stack([pj[..., 2], pj[..., 0], pj[..., 1]], axis=-1)


def draw_skeleton(scene, joints):
    """Add one frame's skeleton to an mjvScene as capsules + spheres."""
    for child, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                            np.zeros(3), np.zeros(3), np.zeros(9), BONE_RGBA)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.014,
                             joints[parent], joints[child])
        scene.ngeom += 1
    for j in joints:
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.022, 0, 0]), j.astype(np.float64),
                            np.eye(3).flatten(), JOINT_RGBA)
        scene.ngeom += 1


def tracking_lookats(track: np.ndarray, alpha=0.12) -> np.ndarray:
    """EMA-smoothed camera lookat targets from a (T, 3) track."""
    out = np.empty_like(track)
    acc = track[0].copy()
    for t, p in enumerate(track):
        acc += alpha * (p - acc)
        out[t] = acc
    out[:, 2] = 0.55  # fixed height: no bobbing when the robot squats
    return out


def scene_model() -> mujoco.MjModel:
    """The run's scene.xml with docs-friendly cosmetics: checkered floor
    (the solver's scene keeps it plain white), light haze, cardboard box."""
    xml = (TASK_DIR / "scene.xml").read_text()
    xml = xml.replace('rgb1="1 1 1" rgb2="1.0 1.0 1.0" markrgb="1.0 1.0 1.0"',
                      'rgb1="0.9 0.9 0.9" rgb2="0.8 0.8 0.8" '
                      'markrgb="0.4 0.4 0.4"')
    xml = xml.replace('haze="0.15 0.25 0.35 1"', 'haze="0.94 0.95 0.97 1"')
    xml = xml.replace('name="largebox_geom" type="box"',
                      'name="largebox_geom" type="box" '
                      'rgba="0.76 0.60 0.42 1"')
    return mujoco.MjModel.from_xml_string(xml)


def make_camera(azimuth=145.0, elevation=-14.0, distance=2.7):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    return cam


def caption(img: np.ndarray, text: str, size=17) -> np.ndarray:
    """Dark caption bar at the bottom of a frame."""
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im, "RGBA")
    bar = 34
    draw.rectangle([0, im.height - bar, im.width, im.height],
                   fill=(20, 24, 32, 235))
    font = ImageFont.truetype(FONT, size)
    tw = draw.textlength(text, font=font)
    draw.text(((im.width - tw) / 2, im.height - bar + (bar - size) / 2 - 2),
              text, font=font, fill=(235, 235, 235))
    return np.asarray(im)


def label(img: np.ndarray, text: str, size=16) -> np.ndarray:
    """Small tag in the top-left corner of a panel."""
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im, "RGBA")
    font = ImageFont.truetype(FONT, size)
    tw = draw.textlength(text, font=font)
    draw.rectangle([10, 10, 10 + tw + 14, 10 + size + 12],
                   fill=(20, 24, 32, 215))
    draw.text((17, 15), text, font=font, fill=(235, 235, 235))
    return np.asarray(im)


def encode_gif(frames, out_path: Path, fps=FPS_OUT):
    """PNG frames -> palette-quantized GIF via ffmpeg."""
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(f"{td}/f_{i:04d}.png")
        filt = ("[0:v]split[a][b];[a]palettegen=max_colors=160:"
                "stats_mode=diff[p];[b][p]paletteuse=dither=bayer:"
                "bayer_scale=5:diff_mode=rectangle")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", f"{td}/f_%04d.png", "-filter_complex", filt,
             str(out_path)], check=True)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


class SceneRenderer:
    def __init__(self, model, width=W, height=H):
        self.model = model
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height, width)

    def frame(self, cam, qpos=None, skeleton=None):
        if qpos is not None:
            self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, cam)
        if skeleton is not None:
            draw_skeleton(self.renderer.scene, skeleton)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "docs")
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    skel = load_skeleton()                              # (228, 34, 3)
    ref = np.load(TASK_DIR / "0/trajectory_kinematic.npz")["qpos"]
    sol = np.load(TASK_DIR / "0/trajectory_mjwp.npz")["qpos"].reshape(-1, 43)
    T = len(ref)
    ts = range(0, T, args.stride)
    lookat = tracking_lookats(ref[:, :3].copy())
    cam = make_camera()

    # -- 1. generation: the raw Kimodo skeleton, nothing else ------------
    stage = SceneRenderer(mujoco.MjModel.from_xml_string(FLOOR_XML))
    gen = []
    for t in ts:
        cam.lookat[:] = lookat[t]
        img = stage.frame(cam, skeleton=skel[t])
        gen.append(caption(img, "“A person lifts a box off the ground "
                                "and carries it forward…”"))
    encode_gif(gen, args.out / "generate.gif")

    # -- 2. recon: the bare clip dissolves into the inferred scene -------
    # (an overlaid skeleton would sit *inside* the opaque G1 meshes, so a
    # crossfade tells the story instead: same motion, scene made explicit)
    scene = SceneRenderer(scene_model())
    HOLD, FADE = 8, 12
    rec = []
    for i, t in enumerate(ts):
        cam.lookat[:] = lookat[t]
        img = scene.frame(cam, qpos=ref[t])
        if i < HOLD + FADE:
            bare = stage.frame(cam, skeleton=skel[t])
            a = max(0.0, (i - HOLD) / FADE)
            img = (bare * (1 - a) + img * a).astype(np.uint8)
        rec.append(caption(img, "hindsight scene recon: retarget to G1, "
                                "infer the box + its placement"))
    encode_gif(rec, args.out / "recon.gif")
    stage.close()

    # -- 3. solve: kinematic reference vs dynamics under mujoco_warp -----
    panel = SceneRenderer(scene.model, width=PANEL_W)
    sim_per_ref = len(sol) // T                         # sim is 60 Hz
    divider = np.full((H, 8, 3), 235, dtype=np.uint8)
    sbs = []
    for t in ts:
        cam.lookat[:] = lookat[t]
        a = label(panel.frame(cam, qpos=ref[t]), "reference (kinematic)")
        b = label(panel.frame(cam, qpos=sol[min(t * sim_per_ref,
                                                len(sol) - 1)]),
                  "solved (SBMPC, dynamic)")
        sbs.append(caption(np.concatenate([a, divider, b], axis=1),
                           "sampling-based MPC → dynamically feasible "
                           "robot + object trajectory"))
    encode_gif(sbs, args.out / "solve.gif")

    scene.close()
    panel.close()


if __name__ == "__main__":
    main()
