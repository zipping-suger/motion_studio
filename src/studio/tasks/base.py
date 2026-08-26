"""The downstream-task interface.

A Task is the MuJoCo-free description of one downstream task: its scene
parameters (what the GUIs and `--set` expose), the choices some of them
take, its solve-parameter deltas, its pass criterion, and a lazy handle
on its reconstruction (`studio.recon.run.ReconTask`, which needs
MuJoCo and is imported only when a reconstruction actually runs). The
solver is the same for every task (studio.solve), so there is nothing
to describe about it here beyond the deltas.

`pipeline`, the CLI and both GUIs drive every task through this and
never branch on a task's name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from ..config import SOLVE_DEFAULTS, TASK_OVERRIDES


def overrides(name: str, section: str) -> dict:
    """config.yml's tasks.<name>.<section> mapping ({} if absent)."""
    return (TASK_OVERRIDES.get(name) or {}).get(section) or {}


def coerce(val, default):
    """A CLI / YAML / GUI value cast to its default's type. Bools accept
    the usual spellings; everything else follows the default's type."""
    if isinstance(default, bool):
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "on")
        return bool(val)
    return type(default)(val)


def _lift(verdict: str) -> bool:
    return verdict == "LIFT"


@dataclass(frozen=True)
class Task:
    name: str
    # the task's scene params; config.yml's `tasks:` section and
    # `--set key=value` override them (scene_params)
    scene_defaults: dict
    # () -> the task's ReconTask; imports the recon (MuJoCo) on call
    recon: Callable[[], object]
    # param -> the tuple of values it takes (a dropdown); other str
    # params are free text, the rest numbers
    choices: dict = field(default_factory=dict)
    # this task's deltas over the shared SOLVE_DEFAULTS
    solve_overrides: dict = field(default_factory=dict)
    # verdict str -> did the run succeed
    passed: Callable[[str], bool] = _lift

    @property
    def solve_defaults(self) -> dict:
        """The shared solve params with this task's deltas applied — what
        the GUIs seed their widgets from."""
        return {**SOLVE_DEFAULTS, **self.solve_overrides}

    def scene_params(self, *layers: Optional[Dict]) -> dict:
        """Defaults + override layers, unknown keys rejected, values
        coerced to the default's type (CLI --set, YAML and the GUIs all
        funnel through here)."""
        out = dict(self.scene_defaults)
        for layer in layers:
            for key, val in (layer or {}).items():
                if key not in out:
                    raise SystemExit(
                        f"unknown {self.name} scene param {key!r} "
                        f"(known: {', '.join(sorted(out))})")
                out[key] = coerce(val, out[key])
        return out

    def solve_params(self, explicit: Optional[Dict] = None) -> dict:
        """The FULLY resolved solve params for one solve: the task's
        defaults, config.yml's tasks.<name>.solve, then what was set
        explicitly. Passed whole to the solve subprocess, which reads no
        config of its own."""
        return {**self.solve_defaults, **overrides(self.name, "solve"),
                **(explicit or {})}

    def reconstruct(self, npz, out_root, name: str,
                    options: Optional[Dict] = None) -> Tuple[Optional[Path], str]:
        """Reconstruct one clip -> (task dir or None, one-line summary).
        `options` carries `scene_params` plus whatever the task's
        detector takes (box_carry: allow_held_start, pick, release)."""
        options = dict(options or {})
        params = self.scene_params(overrides(self.name, "scene"),
                                   options.pop("scene_params", None))
        from ..recon import run  # late: the recon needs MuJoCo
        return run.reconstruct(self.recon(), Path(npz), Path(out_root), name,
                               params, options)
