"""The SBMPC solve: studio's rewards over SPIDER's sampling optimizer.

Split in two by what each half needs to import:

- `evaluate.py` is numpy-only and runs in studio's own venv, so verdicts
  and `studio list` never need the solve runtime.
- `loop.py`, `rewards.py` and `spider_cfg.py` need torch, warp and the
  SPIDER wheel. They run in the solve venv (`studio setup --solve`) as a
  subprocess, reached through `command()` below.

Nothing here imports torch, so the CLI stays instant.
"""

import os
from pathlib import Path

from . import evaluate
from ..config import REPO_ROOT, Config

__all__ = ["evaluate", "available", "require", "command", "env", "MISSING_MSG"]

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
    """Command line for one solve. Run with `env()` as the environment."""
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
