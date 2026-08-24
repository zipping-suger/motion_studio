"""Downstream-task registry.

Each task module exposes a module-level ``TASK`` (see `base.Task`).
Loading is lazy so listing names never drags in a task's heavy imports;
torch never loads here, since solve runtimes are reached only through
each task's subprocess command.

    tasks.load("under_table").reconstruct(npz, out_root, name, options)

config.yml's ``tasks: {<name>: {scene: ..., solve: ...}}`` overrides a
task's defaults per machine; `overrides()` exposes the loaded section so
the CLI and task modules merge from one place.
"""

import importlib

from ..config import TASK_OVERRIDES
from .base import Task

DEFAULT = "box_carry"

_REGISTRY = {
    "box_carry": ".box_carry",
    "under_table": ".under_table",
    "kick": ".kick",
    "ground_pick": ".ground_pick",
    "pole": ".pole",
}


def names() -> list:
    return list(_REGISTRY)


def load(name: str) -> Task:
    if name not in _REGISTRY:
        raise SystemExit(
            f"unknown task {name!r} (available: {', '.join(_REGISTRY)})")
    return importlib.import_module(_REGISTRY[name], __package__).TASK


def overrides(name: str, section: str) -> dict:
    """config.yml's tasks.<name>.<section> mapping ({} if absent)."""
    return (TASK_OVERRIDES.get(name) or {}).get(section) or {}
