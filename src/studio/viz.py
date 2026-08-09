"""A reconstructed scene, drawn into a viser server.

Shared by the two GUIs that render a reconstruction: the solve panel
(scripts/panel_app.py, in the solve venv) and the demo add-on
(scripts/demo_scene_addon.py, in the kimodo venv). Both of those venvs
carry viser + trimesh; the `studio` CLI's own venv does not, so nothing
in the core package may import this module.

A reconstruction is described by the two files `studio.recon` writes into
a task dir: `task_info.json` (box size, terrain geoms, flags) and
`scene.xml` (the MuJoCo scene, which is where the table lives).
"""

import json
import re
from pathlib import Path

import numpy as np
import trimesh

# the table is only in the MuJoCo XML, not in task_info.json
_TABLE_RE = re.compile(r'<geom name="table"[^>]*size="([^"]+)"[^>]*'
                       r'pos="([^"]+)"(?:[^>]*quat="([^"]+)")?')

TABLE_COLOR = (140, 100, 60)
TERRAIN_COLOR = (150, 140, 120)
IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])


def read_info(task_dir: Path) -> dict:
    return json.loads((task_dir / "task_info.json").read_text())


def box_mesh(info: dict):
    return trimesh.creation.box(info["box_size"])


def yaw_wxyz(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def add_scene(server, task_dir: Path, prefix: str = "/scene",
              opacity: float | None = None) -> list:
    """Draw the static reconstruction — table and terrain — under
    `prefix`. Returns the handles, for the caller to remove before the
    next load."""
    info = read_info(task_dir)
    add = server.scene.add_mesh_simple
    # only pass opacity when asked: viser switches to a transparent
    # material as soon as the argument is present
    kw = {} if opacity is None else {"opacity": opacity}
    handles = []

    if info.get("has_table"):
        m = _TABLE_RE.search((task_dir / "scene.xml").read_text())
        if m:
            # MuJoCo box geoms are specified as half-extents
            mesh = trimesh.creation.box(np.fromstring(m.group(1), sep=" ") * 2)
            handles.append(add(
                f"{prefix}/table", mesh.vertices, mesh.faces,
                color=TABLE_COLOR,
                position=np.fromstring(m.group(2), sep=" "),
                wxyz=(np.fromstring(m.group(3), sep=" ") if m.group(3)
                      else IDENTITY_WXYZ),
                **kw))

    for i, geom in enumerate(info.get("terrain_geoms", [])):
        mesh = trimesh.creation.box(np.array(geom["half"]) * 2)
        handles.append(add(
            f"{prefix}/terrain_{i}", mesh.vertices, mesh.faces,
            color=TERRAIN_COLOR, position=np.array(geom["center"]),
            wxyz=yaw_wxyz(geom.get("yaw", 0.0)), **kw))

    return handles


def scene_param_widgets(gui, defaults: dict) -> dict:
    """The scene-reconstruction controls, identical in both GUIs. Returns
    {param name: widget}; read `.value` off each at reconstruct time."""
    widgets = {}
    for key, val in defaults.items():
        if key == "hand_geom":
            widgets[key] = gui.add_dropdown(key, ("mesh", "capsule"),
                                            initial_value=val)
        else:
            widgets[key] = gui.add_number(key, float(val), min=0.0, step=0.005)
    return widgets
