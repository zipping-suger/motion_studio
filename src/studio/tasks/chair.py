"""Chair interaction: lifting, carrying or sitting on a chair around
its real BEHAVE mesh, reconstructed from a clip that carries the
chair's own trajectory (raw_motion/behave).

Contact windows, the sit and the chair's motion are all detected
against the clip's object pose; object="auto" takes the clip's own
object name. The left/right scene params override each hand's contact
window (off / S-E frames). A carried chair lifts (LIFT) or at least
tracks (DR); a sit only tracks.
"""

from ..recon.objects import CHAIR_OBJECTS, GRASPS
from .base import Task

SCENE_DEFAULTS = {
    "object": "auto",     # a measured chair (see CHAIR_OBJECTS) or auto
                          # (the clip's object_name, else its filename)
    "mass": 0.0,          # kg; 0 = the object spec's default
    "grasp_close": 0.0,   # rad: finger-closure override; 0 = calibrated
                          # from the hand model on a chair-member bar
    "grasp": "auto",      # ik_retargeted | reference | auto. Re-aims the
                          # wrist to wrap the chair member under the
                          # palm wherever the hand model exists.
    "left": "auto",       # this hand's GRIP window: auto, off, or
                          # inclusive frames "S-E" / "S-"
    "right": "auto",      # same for the right hand
}
CHOICES = {"object": CHAIR_OBJECTS, "grasp": GRASPS}

# solve deltas over SOLVE_DEFAULTS: the pole's grip weighting (a 3 kg
# chair batted away is as cheap to a low grip term as the lamp was),
# the finger-contact grasp term, and the body-support term that keeps
# a hip or torso carry resting where the reference rested it
SOLVE_OVERRIDES = {"grip_rew_scale": 4.0, "grasp_rew_scale": 0.5,
                   "grasp_pen_rew_scale": 50.0, "support_rew_scale": 0.5}


def _recon():
    from ..recon.chair import TASK
    return TASK


TASK = Task(
    name="chair",
    scene_defaults=SCENE_DEFAULTS,
    recon=_recon,
    choices=CHOICES,
    solve_overrides=SOLVE_OVERRIDES,
    passed=lambda v: v in ("LIFT", "DR"),
)
