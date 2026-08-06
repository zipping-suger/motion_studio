"""Load config.yml and derive the paths every command needs."""

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"

# mppi_locoma's trial subtree (mirrors its repo_config.TASKS_DIR layout);
# trials of run <name> live at <run>/outputs/<TASK_SUBTREE>/<task>/.
TASK_SUBTREE = Path("processed/kimodo/unitree_g1/humanoid_object")


@dataclass
class Config:
    kimodo_repo: Path
    kimodo_python: Path
    mppi_locoma: Path
    model: str
    demo_port: int
    encoder_host: str
    encoder_port: int

    @property
    def mppi_python(self) -> Path:
        return self.mppi_locoma / ".venv/bin/python"

    @property
    def examples_dir(self) -> Path:
        # The demo's hardcoded Save Example base: <pkg>/assets/demo/examples/
        # <short_key> (kimodo/assets.py + kimodo/demo/config.py). The short
        # key is the display name lowercased minus the version suffix,
        # e.g. Kimodo-G1-RP-v1 -> kimodo-g1-rp.
        short = self.model.lower().removesuffix("-v1")
        return self.kimodo_repo / "kimodo/assets/demo/examples" / short


def load_config() -> Config:
    cfg = yaml.safe_load((REPO_ROOT / "config.yml").read_text())
    paths = cfg["paths"]
    return Config(
        kimodo_repo=Path(paths["kimodo_repo"]),
        kimodo_python=Path(paths["kimodo_python"]),
        mppi_locoma=Path(paths["mppi_locoma"]),
        model=cfg["demo"]["model"],
        demo_port=int(cfg["demo"]["port"]),
        encoder_host=cfg["encoder"]["remote_host"],
        encoder_port=int(cfg["encoder"]["port"]),
    )
