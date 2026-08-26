"""Load config.yml and derive the paths every command needs."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .solve.defaults import SOLVE_DEFAULTS, SOLVE_INT_KEYS  # noqa: F401

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

# box_carry's scene-reconstruction parameters, as every GUI exposes them
# (the other tasks' live with the task, studio.tasks.<task>.SCENE_DEFAULTS;
# all of them are MuJoCo-free so the demo add-on can read them).
# Overridable from config.yml's `scene:` section and per-run from the
# GUIs and `studio recon` flags.
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

# config.yml's `tasks:` section, e.g.
# tasks: {pole: {scene: {object: tripod}, solve: {...}}}; the top-level
# scene:/solve: sections above are box_carry's equivalents.
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
