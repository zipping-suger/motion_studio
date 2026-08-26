"""The object roster: what a task may reconstruct around.

Small primitives (a one-hand pick) come from `SHAPES`; measured meshes
come from the sidecars under assets/object_mesh/ (`scripts/
prep_object_mesh.py` for OMOMO poles, behave_pipeline's export for
chairs) — one JSON per object next to its .obj and convex hulls. This
module is MuJoCo-free on purpose: the GUIs (including the demo add-on
in the kimodo venv) read the rosters to build their dropdowns, and the
task modules read them without dragging the recon in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from ..config import REPO_ROOT

OBJECT_MESH_DIR = REPO_ROOT / "assets" / "object_mesh"

# ground_pick's primitives: a cube edge / cylinder diameter / ball diameter
SHAPES = ("cube", "cylinder", "ball")
# how a reconstruction poses the hand: the clip's own wrist, or re-aimed
# by IK to wrap the estimated object ("auto" lets the task choose)
GRASPS = ("auto", "reference", "ik_retargeted")


def sidecars() -> Dict[str, dict]:
    """name -> parsed sidecar, for every readable JSON under the dir."""
    out = {}
    for p in sorted(OBJECT_MESH_DIR.glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text())
        except ValueError:
            continue
    return out


def pole_objects() -> Tuple[str, ...]:
    """The pole-like objects: a handle_z band marks one."""
    return tuple(n for n, d in sidecars().items() if "handle_z" in d)


def chair_objects() -> Tuple[str, ...]:
    return tuple(n for n, d in sidecars().items()
                 if d.get("category") == "chair")


# the rosters as the GUIs list them ("auto" = from the clip)
POLE_OBJECTS = ("auto",) + pole_objects()
CHAIR_OBJECTS = ("auto",) + chair_objects()


def load_sidecar(name: str, what: str = "object") -> dict:
    path = OBJECT_MESH_DIR / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"no {what} spec {path}")
    return json.loads(path.read_text())


def infer_by_name(clip_name: str, keywords: Sequence[Tuple[str, str]],
                  roster: Sequence[str], default: Optional[str]) -> Optional[str]:
    """The roster entry a clip's name asks for: the first keyword hit whose
    object is actually present, else ``default``."""
    name = clip_name.lower()
    for key, obj in keywords:
        if key in name and obj in roster:
            return obj
    return default


def sidecar_path(name: str) -> Path:
    return OBJECT_MESH_DIR / f"{name}.json"
