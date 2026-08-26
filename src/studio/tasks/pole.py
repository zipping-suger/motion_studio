"""Pole interaction: holding or carrying a pole-like object (floor
lamp, tripod, clothes stand) around its real OMOMO mesh, reconstructed
from a hold clip.

The hold window, hands, per-hand engagement and wrist retarget are all
detected; object="auto" picks the mesh from the clip's filename. The
left/right scene params override each hand's contact window (off / S-E
frames) — e.g. carry a lamp one-handed from a two-hand clip. A carried
pole barely rises, so DR tracking success passes too, with LIFT still
counting when a clip does lift.
"""

from ..recon.objects import GRASPS, POLE_OBJECTS
from .base import Task

# the window, hands, engagement and wrist retarget are all detected
# from the clip, so these are the only knobs
SCENE_DEFAULTS = {
    "object": "auto",     # a measured mesh (see POLE_OBJECTS) or auto
                          # (from the clip name)
    "mass": 0.0,          # kg; 0 = the object spec's default
    "grasp_close": 0.0,   # rad: wrap-closure override; 0 = calibrated
                          # from the hand model to touch the handle
    "grasp": "auto",      # ik_retargeted | reference | auto. Re-aims the
                          # wrist to wrap the pole's axis wherever the
                          # hand model exists.
    "left": "auto",       # this hand's contact window: auto, off, or
                          # inclusive frames "S-E" / "S-"
    "right": "auto",      # same for the right hand
}
CHOICES = {"object": POLE_OBJECTS, "grasp": GRASPS}

# solve deltas over SOLVE_DEFAULTS. At grip 1.0 the 8.0 object
# tracking term makes BATTING the pole cheaper than holding it: a solve
# kept hands on only 53% of frames and dropped it during the end pad.
# At 4.0 the same solve held every frame, object error 0.176 -> 0.054.
SOLVE_OVERRIDES = {"grip_rew_scale": 4.0}


def _recon():
    from ..recon.pole import TASK
    return TASK


TASK = Task(
    name="pole",
    scene_defaults=SCENE_DEFAULTS,
    recon=_recon,
    choices=CHOICES,
    solve_overrides=SOLVE_OVERRIDES,
    passed=lambda v: v in ("LIFT", "DR"),
)
