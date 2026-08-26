"""Downstream-task registry.

Each task module exposes a module-level ``TASK`` (see `base.Task`) and
is MuJoCo-free: a task's reconstruction is imported only when it runs,
so listing tasks and reading their parameters works in every venv
(the demo add-on builds its dropdowns from these).

    tasks.load("pole").reconstruct(npz, out_root, name, options)

config.yml's ``tasks: {<name>: {scene: ..., solve: ...}}`` overrides a
task's defaults per machine; `overrides()` exposes the loaded section so
the CLI and task modules merge from one place.
"""

import importlib

from .base import Task, overrides

__all__ = ["DEFAULT", "Task", "load", "names", "overrides"]

DEFAULT = "box_carry"

_REGISTRY = {
    "box_carry": ".box_carry",
    "ground_pick": ".ground_pick",
    "pole": ".pole",
    "chair": ".chair",
}


def names() -> list:
    return list(_REGISTRY)


def load(name: str) -> Task:
    if name not in _REGISTRY:
        raise SystemExit(
            f"unknown task {name!r} (available: {', '.join(_REGISTRY)})")
    return importlib.import_module(_REGISTRY[name], __package__).TASK
