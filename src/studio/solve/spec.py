"""What the solver reads from a trial's task_info.json — the recon -> solve
contract, in one place and without task names.

Every downstream task is a human-object interaction: a robot, one free
object, optional static terrain. What differs between tasks reaches the
solver as DATA written by `studio.recon.spec.write_trial`:

- ``object.faces``     the object is a box whose two palm-side faces are
                       the contact targets (the face-contact reward);
                       otherwise the baked palm tracks are
- ``object.symmetry``  "axial": the object's roll about its own axis is
                       meaningless (a capsule handle), so the evaluator
                       scores only the axis tilt
- ``grips``            per holding hand: its calibrated palm pocket and
                       the object-frame anchor the grip reward pulls into
                       it (anchor null = the object's origin)
- ``supports``         body geoms the reference rests the object on,
                       with their windows (the support reward)
- ``pick_frame`` / ``release_frame``   the interaction window (object
                       tracking weight profile)
- ``hand_noise_scale`` finger-actuator exploration damping

numpy-free, so the studio venv's evaluator reads it too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PALM_COLUMN = {"left": 0, "right": 1}


@dataclass
class SolveScene:
    faces: bool = False
    axial: bool = False
    # (pocket xyz in the palm-site frame, palm column, anchor xyz | None)
    grips: list = field(default_factory=list)
    supports: list = field(default_factory=list)
    hand_noise_scale: Optional[float] = None
    pick_frame: Optional[int] = None
    release_frame: Optional[int] = None
    ref_dt: Optional[float] = None
    quality_flags: list = field(default_factory=list)

    @classmethod
    def from_info(cls, info: dict) -> "SolveScene":
        obj = info.get("object")
        grips = [(g["pocket"], PALM_COLUMN[g["hand"]], g.get("anchor"))
                 for g in info.get("grips") or []
                 if g.get("pocket") is not None]
        if obj is None:
            # trials written before task_info carried `object` (the box
            # recon wrote box_size at the top level and declared nothing
            # else): the only such trials are boxes, whose faces are the
            # contact targets
            obj = {"faces": "box_size" in info}
        if "grips" not in info and info.get("grasp_pocket") is not None:
            # likewise a pre-spec one-hand pick: its pocket and hand sat
            # at the top level
            grips = [(info["grasp_pocket"], PALM_COLUMN[info["pick_hand"]],
                      None)]
        axial = obj.get("symmetry") == "axial" or (
            "symmetry" not in obj and "handle_radius" in obj)  # pre-spec pole
        return cls(
            faces=bool(obj.get("faces", False)),
            axial=axial,
            grips=grips,
            supports=list(info.get("supports") or []),
            hand_noise_scale=info.get("hand_noise_scale"),
            pick_frame=info.get("pick_frame"),
            release_frame=info.get("release_frame"),
            ref_dt=info.get("ref_dt"),
            quality_flags=list(info.get("quality_flags") or []),
        )

    @classmethod
    def load(cls, path: Path) -> "SolveScene":
        """From a task_info.json; an absent file is an empty scene."""
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.from_info(json.loads(path.read_text()))
