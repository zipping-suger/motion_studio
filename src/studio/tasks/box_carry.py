"""The original task: hindsight box reconstruction (SceneBot-style) +
the SBMPC solve.

The scene defaults are the very dict config.load_config() mutates, so
config.yml's top-level ``scene:`` section keeps working unchanged; the
solve defaults are the shared ones as they are. A run passes on LIFT.
"""

from ..config import SCENE_DEFAULTS
from .base import Task

CHOICES = {"hand_geom": ("mesh", "capsule")}


def _recon():
    from ..recon.box_carry import TASK
    return TASK


TASK = Task(
    name="box_carry",
    scene_defaults=SCENE_DEFAULTS,
    recon=_recon,
    choices=CHOICES,
)
