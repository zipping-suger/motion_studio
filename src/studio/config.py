"""Load config.yml and derive the paths every command needs."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"
RAW_MOTION_DIR = REPO_ROOT / "raw_motion"

# where `studio setup` puts the two heavy runtimes: third-party stacks
# with their own pinned torch, kept out of studio's own venv
DEFAULT_KIMODO_PYTHON = REPO_ROOT / ".venv-kimodo/bin/python"
DEFAULT_SOLVE_PYTHON = REPO_ROOT / ".venv-solve/bin/python"

# the pinned upstream checkouts (git submodules), initialized on demand by
# `studio setup`; config.yml's paths section points elsewhere, e.g. a fork
DEFAULT_KIMODO_REPO = REPO_ROOT / "extern/kimodo"
DEFAULT_KIMODO_VISER = REPO_ROOT / "extern/kimodo-viser"
DEFAULT_SPIDER = REPO_ROOT / "extern/spider"

# SPIDER's trial layout: trials of run <name> live under
# <run>/outputs/<PROCESSED_ROOT>/<embodiment>/<task>/, the embodiment
# segment varying per task. TASK_SUBTREE is the box-carry one, which
# SPIDER's own dataset resolution must keep matching.
PROCESSED_ROOT = Path("processed/kimodo/unitree_g1")
TASK_SUBTREE = PROCESSED_ROOT / "humanoid_object"

# Scene-reconstruction parameters, as all three GUIs expose them. Here
# rather than in studio.recon because the demo add-on runs in the kimodo
# venv, which has no MuJoCo. Overridable from config.yml's `scene:`
# section and per-run from the GUIs and `studio recon` flags.
SCENE_DEFAULTS = {
    "box_mass": 0.2,        # kg
    "box_height": 0.20,     # m
    "box_depth": 0.24,      # m
    "squeeze": 0.008,       # m, extra box width so reference hands press in
    "hand_geom": "mesh",    # mesh (rubber-hand convex hull) | capsule
    # below this support height, extend the box to the floor instead of
    # building a slab, palms gripping at its centroid
    "floor_snap_below": 0.25,
}

# ground_pick's SCENE_DEFAULTS, here for the same reason as above.
# Overridable from config.yml's `tasks: {ground_pick: {scene: ...}}` and
# per-run from the GUIs and `studio recon --set`.
PICK_SCENE_DEFAULTS = {
    "shape": "auto",      # cube | cylinder | ball | auto (from clip name)
    "size": 0.06,         # m: cube edge / cylinder diameter / ball diameter
    "cyl_height": 0.10,   # m: cylinder height. Keep it under the hand's
                          # floor-reach height — a taller bottle's top
                          # sits above the palm-down approach.
    "mass": 0.10,         # kg
    "grasp_close": 0.0,   # rad: wrap-closure override; 0 = calibrated
                          # from the hand model to touch the object
    "grasp": "auto",      # reference | ik_retargeted | auto. Cylinders
                          # get ik_retargeted: the wrist is re-aimed so
                          # the palm wraps the curved surface.
}
PICK_SHAPES = ("auto", "cube", "cylinder", "ball")
PICK_GRASPS = ("auto", "reference", "ik_retargeted")

# the pole task's SCENE_DEFAULTS, here for the same reason as above. The
# window, hands, engagement and wrist retarget are all detected from the
# clip, so these are the only knobs.
POLE_SCENE_DEFAULTS = {
    "object": "auto",     # a measured mesh (see POLE_OBJECTS) or auto
                          # (from the clip name)
    "mass": 0.0,          # kg; 0 = the object spec's default
    "grasp_close": 0.0,   # rad: wrap-closure override; 0 = calibrated
                          # from the hand model to touch the handle
    "grasp": "auto",      # ik_retargeted | reference | auto. Re-aims the
                          # wrist to wrap the pole's axis wherever the
                          # hand model exists.
    "left": "auto",       # this hand's contact window: auto, off, or
                          # inclusive frames "S-E" / "S-"
    "right": "auto",      # same for the right hand
}
# the roster is whatever sidecars exist under assets/object_mesh/
POLE_OBJECTS = ("auto",) + tuple(sorted(
    p.stem for p in (REPO_ROOT / "assets" / "object_mesh").glob("*.json")))

# pole solve deltas over SOLVE_DEFAULTS. At grip 1.0 the 8.0 object
# tracking term makes BATTING the pole cheaper than holding it: a solve
# kept hands on only 53% of frames and dropped it during the end pad.
# At 4.0 the same solve held every frame, object error 0.176 -> 0.054.
POLE_SOLVE_OVERRIDES = {"grip_rew_scale": 4.0}

# SBMPC solve hyperparameters, as the panel exposes them: object
# interaction is the objective, robot tracking only guides. The fixed task
# setup lives in studio.solve.spider_cfg. Overridable from config.yml's
# `solve:` section.
SOLVE_DEFAULTS = {
    "num_samples": 2048,        # parallel mujoco_warp worlds (VRAM-bound)
    "horizon": 1.0,             # s, planning horizon
    "max_num_iterations": 16,   # annealing iterations per control tick
    "temperature": 1.0,         # softmax temperature over the top 10%
    "joint_rew_scale": 0.5,     # robot joints (guide only)
    "base_pos_rew_scale": 1.0,
    "base_rot_rew_scale": 1.0,
    "pos_rew_scale": 8.0,       # object position (the objective)
    "rot_rew_scale": 0.2,       # object orientation (barely matters for a box)
    "contact_rew_scale": 2.0,   # pulls each holding palm onto the object
                                # through its hold window, so intermittent
                                # slap-and-release prices worse than a
                                # maintained hold. box_carry uses the
                                # simulated box's grasp faces, the others
                                # the baked palm track.
    "vel_rew_scale": 1.0,       # master scale of the per-block velocity term
    "grip_rew_scale": 1.0,      # keep the SIMULATED object in the palm's
                                # calibrated pocket: slip shows as
                                # palm-relative drift long before a drop
    "upright_rew_scale": 10.0,  # anti-fall: penalty for tilting PAST the
                                # reference's own tilt or sinking BELOW
                                # its height, so bending clips stay free.
                                # Tracking alone prices a fall as another
                                # pose error, and MPPI takes that trade.
    "obj_end_rew_scale": 1.5,   # object-tracking multiplier at the
                                # interaction's endpoints, which ARE the
                                # objective...
    "obj_mid_rew_scale": 0.5,   # ...and mid-interaction, where the
                                # carried path matters less
    "self_collision_rew_scale": 50.0,  # penalty per metre of summed
                                # penetration across the template's ghost
                                # self-pairs. They exert no force, so this
                                # term is what steers rollouts apart.
}

# solve params that must stay integers when they reach the optimizer
SOLVE_INT_KEYS = frozenset({"num_samples", "max_num_iterations"})

# config.yml's `tasks:` section, e.g.
# tasks: {under_table: {scene: {yaw_range: 0}, solve: {...}}}. The dicts
# above are box_carry's equivalents, kept at top level for compatibility.
TASK_OVERRIDES: dict = {}


def config_path() -> Path:
    """config.yml next to the repo, or wherever STUDIO_CONFIG points."""
    override = os.environ.get("STUDIO_CONFIG")
    return Path(override).expanduser() if override else REPO_ROOT / "config.yml"


def _path(value: str) -> Path:
    """Paths in config.yml may use ~ and $VARS."""
    return Path(os.path.expandvars(str(value))).expanduser()


@dataclass
class Config:
    kimodo_repo: Path
    kimodo_python: Path
    solve_python: Path
    model: str
    demo_port: int
    encoder_host: str
    encoder_port: int
    # a SPIDER checkout or wheel, needed once by `studio setup` for the G1
    # assets and the solve venv; never at run time
    spider: Path = DEFAULT_SPIDER
    # a unitree_g1 checkout with the BrainCo-hand URDF + meshes, read once
    # by `studio setup --assets`; without it the handless model is used
    unitree_g1: Path | None = None
    # where `studio promote` copies passing trials
    dataset_outputs: Path | None = None

    @property
    def model_short(self) -> str:
        # registry short key, e.g. Kimodo-G1-RP-v1 -> kimodo-g1-rp; also
        # the demo's embedding-cache namespace
        return self.model.lower().removesuffix("-v1")

    @property
    def examples_dir(self) -> Path:
        # the demo's hardcoded Save Example base
        return self.kimodo_repo / "kimodo/assets/demo/examples" / self.model_short


def load_config() -> Config:
    """config.yml is optional — every value has a default, so a fresh
    clone works with no config at all. The file only overrides."""
    path = config_path()
    cfg = (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}
    SCENE_DEFAULTS.update(cfg.get("scene") or {})
    SOLVE_DEFAULTS.update(cfg.get("solve") or {})
    TASK_OVERRIDES.clear()
    TASK_OVERRIDES.update(cfg.get("tasks") or {})
    paths = cfg.get("paths") or {}
    demo = cfg.get("demo") or {}
    encoder = cfg.get("encoder") or {}

    def opt(key: str, default):
        return _path(paths[key]) if paths.get(key) else default

    return Config(
        kimodo_repo=opt("kimodo_repo", DEFAULT_KIMODO_REPO),
        kimodo_python=opt("kimodo_python", DEFAULT_KIMODO_PYTHON),
        solve_python=opt("solve_python", DEFAULT_SOLVE_PYTHON),
        spider=opt("spider", DEFAULT_SPIDER),
        unitree_g1=opt("unitree_g1", None),
        dataset_outputs=opt("dataset_outputs", None),
        model=demo.get("model", "Kimodo-G1-RP-v1"),
        demo_port=int(demo.get("port", 7860)),
        encoder_host=encoder.get("remote_host", "euler"),
        encoder_port=int(encoder.get("port", 9550)),
    )


# ------------------------------------------------------------- runs --

def task_root(run_dir: Path) -> Path:
    """Where a run's built box-carry trials live."""
    return run_dir / "outputs" / TASK_SUBTREE


def task_dirs(run_dir: Path, prefix: str = "") -> list[Path]:
    """A run's built trial dirs across every task's embodiment subtree
    (empty if never reconstructed). A run holds one task type, so the
    union is that task's trials."""
    root = run_dir / "outputs" / PROCESSED_ROOT
    if not root.is_dir():
        return []
    return sorted((d for emb in root.iterdir() if emb.is_dir()
                   for d in emb.iterdir()
                   if d.is_dir() and d.name.startswith(prefix)),
                  key=lambda d: d.name)
