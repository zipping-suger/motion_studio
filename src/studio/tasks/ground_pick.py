"""Ground pick: one-hand pick-up of a small object (cube / cylinder /
ball) from the floor, reconstructed from a single-hand reach clip.

shape="auto" reads cube/cylinder/ball from the clip's filename; a run
holds one shape, so reconstruct again with --set shape=... for others.
The trial declares one grip and no box faces, which is all the solve
needs; a run passes on LIFT.
"""

from ..recon.objects import GRASPS, SHAPES
from .base import Task

SCENE_DEFAULTS = {
    "shape": "auto",      # cube | cylinder | ball | auto (from clip name)
    "size": 0.06,         # m: cube edge / cylinder diameter / ball diameter
    "cyl_height": 0.10,   # m: cylinder height. Keep it under the hand's
                          # floor-reach height — a taller bottle's top
                          # sits above the palm-down approach.
    "mass": 0.10,         # kg
    "grasp_close": 0.0,   # rad: wrap-closure override; 0 = calibrated
                          # from the hand model to touch the object
    "grasp": "auto",      # reference | ik_retargeted | auto. Cylinders
                          # get ik_retargeted: the wrist is re-aimed so
                          # the palm wraps the curved surface.
}
CHOICES = {"shape": ("auto",) + SHAPES, "grasp": GRASPS}


def _recon():
    from ..recon.pick import TASK
    return TASK


TASK = Task(
    name="ground_pick",
    scene_defaults=SCENE_DEFAULTS,
    recon=_recon,
    choices=CHOICES,
)
