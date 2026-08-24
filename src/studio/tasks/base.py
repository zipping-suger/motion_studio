"""The downstream-task interface.

A Task bundles what varies between downstream tasks — how a clip becomes
a scene, how it is solved, how the result is judged — so `pipeline` and
the CLI stay task-agnostic. Everything else is shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Task:
    name: str
    # trial layout under a run's outputs/ (SPIDER's processed-data subtree;
    # the embodiment segment differs per task)
    subtree: Path
    # params the CLI/GUIs expose; config.yml's `tasks:` section and
    # `--set key=value` override them (see tasks.overrides)
    scene_defaults: dict
    solve_defaults: dict
    solve_int_keys: frozenset
    # (npz, out_root, task_name, options) -> (task_dir | None, summary line)
    # None task_dir = the motion admits no scene of this family (a
    # verdict); raise for real errors.
    reconstruct: Callable
    # (cfg, task_name, dataset_dir, params) -> subprocess argv for one solve
    solve_command: Callable
    # (task_dirs) -> [(task_name, metrics dict | None)]; numpy/mujoco only,
    # since verdicts must never need the solve venv
    evaluate: Callable
    # rows -> printable table / (name, metrics) -> one line / rows -> verdict
    format_table: Callable
    format_row: Callable
    verdict: Callable
    # verdict str -> did the run succeed
    passed: Callable
