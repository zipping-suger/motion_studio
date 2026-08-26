"""Launching the solve runtime from studio's own venv.

The solve needs torch + warp + SPIDER, which live in a separate venv
(`config.Config.solve_python`). This module is the studio-side half of
that boundary: is the venv there, and what command + environment run
`studio.solve.loop` in it. The solve side (`studio.solve`) imports
nothing from here, or from `studio.config` — it only sees the command
line built below.
"""

import os
from pathlib import Path

from .config import REPO_ROOT, Config

MISSING_MSG = (
    "the solve venv is missing.\n"
    "  uv run studio setup --solve    # torch + warp + the SPIDER wheel"
)


def available(cfg: Config) -> bool:
    return cfg.solve_python.exists()


def require(cfg: Config) -> None:
    if not available(cfg):
        raise SystemExit(MISSING_MSG)


def command(cfg: Config, task: str, dataset_dir: Path,
            params: dict | None = None) -> list[str]:
    """Command line for one solve. Run with `env()` as the environment.

    ``params`` should be the FULLY resolved solve params (see
    `studio.tasks.Task.solve_params`): the subprocess never reads
    config.yml or a task's defaults, only what it is handed here."""
    require(cfg)
    args = [str(cfg.solve_python), "-m", "studio.solve.loop",
            "--task", task, "--dataset-dir", str(dataset_dir)]
    for key, value in (params or {}).items():
        args += ["--param", f"{key}={value}"]
    return args


def env() -> dict:
    """The solve venv has no `studio` installed — point it at our source.
    MUJOCO_GL=egl keeps warp/mujoco headless."""
    return {
        "PYTHONPATH": os.pathsep.join(
            [str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
        "MUJOCO_GL": "egl",
    }
