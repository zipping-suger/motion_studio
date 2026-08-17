"""Load config.yml and derive the paths every command needs."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"
RAW_MOTION_DIR = REPO_ROOT / "raw_motion"

# where `studio setup` puts the two heavy runtimes unless config.yml
# overrides them. Each is a third-party stack with its own pinned torch, so
# they stay out of studio's own venv.
DEFAULT_KIMODO_PYTHON = REPO_ROOT / ".venv-kimodo/bin/python"
DEFAULT_SOLVE_PYTHON = REPO_ROOT / ".venv-solve/bin/python"

# the pinned upstream checkouts (git submodules). `studio setup` initializes
# them on demand; config.yml's paths section overrides them with checkouts
# you manage elsewhere (e.g. a fork you hack on).
DEFAULT_KIMODO_REPO = REPO_ROOT / "extern/kimodo"
DEFAULT_KIMODO_VISER = REPO_ROOT / "extern/kimodo-viser"
DEFAULT_SPIDER = REPO_ROOT / "extern/spider"

# SPIDER's trial layout (spider.io.get_processed_data_dir); trials of run
# <name> live at <run>/outputs/<PROCESSED_ROOT>/<embodiment>/<task>/. The
# embodiment segment is per downstream task (box carry tracks an object,
# obstacle tasks are robot-only); TASK_SUBTREE is the box-carry one, which
# SPIDER's own dataset resolution must keep matching.
PROCESSED_ROOT = Path("processed/kimodo/unitree_g1")
TASK_SUBTREE = PROCESSED_ROOT / "humanoid_object"

# Scene-reconstruction parameters, as all three GUIs expose them.
# They live here, not in studio.recon, because the demo add-on runs in the
# kimodo venv — which has no MuJoCo and so cannot import that package.
# Overridable per-machine from config.yml's optional `scene:` section, and
# per-run from the panel, the demo add-on, and `studio recon` flags.
SCENE_DEFAULTS = {
    "box_mass": 0.2,        # kg
    "box_height": 0.20,     # m
    "box_depth": 0.24,      # m
    "squeeze": 0.008,       # m, extra box width so reference hands press in
    "hand_geom": "mesh",    # mesh (rubber-hand convex hull) | capsule
    # if the inferred support surface is below this height, extend the box
    # down to rest on the floor (no slab) with palms gripping at its centroid
    "floor_snap_below": 0.25,
}

# SBMPC solve hyperparameters, as the panel exposes them. Object interaction
# is the objective; robot tracking only guides. Everything else about the
# solve is fixed task setup and lives in studio.solve.spider_cfg.
# Overridable from config.yml's optional `solve:` section.
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
    "contact_rew_scale": 1.0,   # palm -> box-face contact reward
    "vel_rew_scale": 1.0,       # master scale of the per-block velocity term
}

# solve params that must stay integers when they reach the optimizer
SOLVE_INT_KEYS = frozenset({"num_samples", "max_num_iterations"})

# config.yml's optional `tasks:` section — per-task defaults overrides,
# e.g. tasks: {under_table: {scene: {yaw_range: 0}, solve: {...}}}. The
# dicts above are the box_carry task's equivalents, kept at top level for
# backward compatibility. Consumed via studio.tasks.overrides().
TASK_OVERRIDES: dict = {}


def config_path() -> Path:
    """config.yml next to the repo, or wherever STUDIO_CONFIG points."""
    override = os.environ.get("STUDIO_CONFIG")
    return Path(override).expanduser() if override else REPO_ROOT / "config.yml"


def _path(value: str) -> Path:
    """Paths in config.yml may use ~ and $VARS, so one checked-in example
    can serve machines with different layouts."""
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
    # A SPIDER checkout or wheel. Needed once by `studio setup` — for the G1
    # assets and to build the solve venv — never at run time.
    spider: Path = DEFAULT_SPIDER
    # Where `studio promote` copies passing trials, if you keep a central
    # dataset elsewhere.
    dataset_outputs: Path | None = None

    @property
    def model_short(self) -> str:
        # Registry short key: display name lowercased minus the version
        # suffix, e.g. Kimodo-G1-RP-v1 -> kimodo-g1-rp. Also the demo's
        # embedding-cache namespace.
        return self.model.lower().removesuffix("-v1")

    @property
    def examples_dir(self) -> Path:
        # The demo's hardcoded Save Example base: <pkg>/assets/demo/examples/
        # <short_key> (kimodo/assets.py + kimodo/demo/config.py).
        return self.kimodo_repo / "kimodo/assets/demo/examples" / self.model_short


def load_config() -> Config:
    """config.yml is optional: every value has a default (the external repos
    default to the extern/ submodules), so a fresh clone works with no config
    at all. The file only overrides."""
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
    """A run's built trial dirs, across every task's embodiment subtree
    (empty if it was never reconstructed). A run only ever holds one task
    type, so the union is that task's trials."""
    root = run_dir / "outputs" / PROCESSED_ROOT
    if not root.is_dir():
        return []
    return sorted((d for emb in root.iterdir() if emb.is_dir()
                   for d in emb.iterdir()
                   if d.is_dir() and d.name.startswith(prefix)),
                  key=lambda d: d.name)
