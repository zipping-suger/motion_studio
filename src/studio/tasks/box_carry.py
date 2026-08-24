"""The original task, wrapped: hindsight box reconstruction + SBMPC solve.

A thin adapter onto the Task interface; the science lives in
studio.recon / studio.solve. The defaults are the very dicts
config.load_config() mutates, so config.yml's top-level ``scene:`` /
``solve:`` sections keep working unchanged.
"""

from .. import recon, solve
from ..config import SCENE_DEFAULTS, SOLVE_DEFAULTS, SOLVE_INT_KEYS, TASK_SUBTREE
from .base import Task


def _reconstruct(npz, out_root, task, options):
    task_dir, _grasp, line = recon.reconstruct(npz, out_root, task,
                                               **(options or {}))
    return task_dir, line


TASK = Task(
    name="box_carry",
    subtree=TASK_SUBTREE,
    scene_defaults=SCENE_DEFAULTS,
    solve_defaults=SOLVE_DEFAULTS,
    solve_int_keys=SOLVE_INT_KEYS,
    reconstruct=_reconstruct,
    solve_command=solve.command,
    evaluate=solve.evaluate.evaluate_tasks,
    format_table=solve.evaluate.format_table,
    format_row=solve.evaluate.format_row,
    verdict=solve.evaluate.verdict,
    passed=lambda v: v == "LIFT",
)
