"""box_carry as a ReconTask (recon.run): the hindsight box + terrain
reconstruction of `scene.py`, driven by `grasp.py`'s bimanual detector.

The box is the only object whose palm-side faces are the contact
targets (`object.faces`), so the solve's face-contact reward turns on
for it and the grip reward stays off (no grips are declared: the
reference's palms squeeze the box on purpose, see grasp.HAND_SURFACE_
OFFSET). The interaction graph is built here rather than by the driver
because the terrain reconstruction consumes it first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import mujoco
import numpy as np

from . import layout
from .graph import build_interaction_graph
from .grasp import GraspInfo, detect_grasp, override_window
from .loader import load_kimodo_constraints
from .scene import (build_object_trajectory, generate_scene_xml,
                    reconstruct_terrain)
from .spec import (Built, Interaction, ObjectSpec, Skip, TrialSpec,
                   assemble_qpos)


class BoxCarryTask:
    name = "box_carry"

    def detect(self, qpos, meta, params, options) -> Interaction:
        """The grasp window; `options` may force it (pick / release
        frames, either alone keeping the detected other) or let the
        detector accept a clip that begins mid-hold."""
        npz = Path(meta["file_path"])
        cons = load_kimodo_constraints(npz)
        grasp = detect_grasp(meta, qpos, cons,
                             allow_held_start=bool(
                                 options.get("allow_held_start", False)))
        pick, release = options.get("pick"), options.get("release")
        if pick is not None or release is not None:
            grasp = override_window(
                meta, grasp,
                grasp.pick_frame if pick is None else pick,
                grasp.release_frame if release is None else release)
        detail = (f"pick f{grasp.pick_frame} release f{grasp.release_frame} "
                  f"box_w {grasp.box_width:.3f} (raw {grasp.raw_carry_gap:.3f}) "
                  f"lift {grasp.lift_height:.2f} "
                  f"flags [{','.join(grasp.quality_flags) or '-'}]")
        return Interaction(True, detail, info=grasp)

    def build(self, qpos_robot, meta, inter: Interaction, params) -> Built:
        grasp: GraspInfo = inter.info
        obj_qpos, spec = build_object_trajectory(
            meta, grasp,
            box_mass=float(params["box_mass"]),
            box_height=float(params["box_height"]),
            box_depth=float(params["box_depth"]),
            squeeze=float(params["squeeze"]),
            floor_snap_below=float(params["floor_snap_below"]))
        graph = build_interaction_graph(meta, grasp)
        terr = reconstruct_terrain(meta, graph, spec)
        if terr is None:
            raise Skip("motion inconsistent with any scene: the robot "
                       "sweeps through the box's resting support")
        spec.plateaus, terrain_stats = terr

        scene_xml = generate_scene_xml(spec, hand_geom=str(params["hand_geom"]))
        model = mujoco.MjModel.from_xml_string(scene_xml)
        layout.check_scene(model)
        qpos = assemble_qpos(model, qpos_robot, obj_qpos)

        # contact reference for the solver: over the grasp window, pull
        # the palm sites onto the box side-face centers, so holding the
        # box is rewarded directly rather than through pose mimicry
        T = len(qpos)
        contact = np.zeros((T, 2))
        contact_pos = np.zeros((T, 2, 3))
        contact[grasp.pick_frame:grasp.release_frame + 1] = 1.0
        for t in range(T):
            box_p, box_q = qpos[t, -7:-4], qpos[t, -4:]
            yaw = 2 * np.arctan2(box_q[3], box_q[0])
            face = np.array([np.cos(yaw), np.sin(yaw), 0.0]) * spec.box.half_w
            contact_pos[t, 0] = box_p + face   # left palm (+x_box side)
            contact_pos[t, 1] = box_p - face   # right palm

        b = spec.box
        tspec = TrialSpec(
            task_type="box_carry",
            object=ObjectSpec(kind="box",
                              size=[2 * b.half_w, 2 * b.half_d, 2 * b.half_h],
                              mass=b.mass, faces=True),
            window=(grasp.pick_frame, grasp.release_frame),
            starts_held=bool(grasp.starts_held),
            lift_height=grasp.lift_height,
            flags=list(grasp.quality_flags),
            graph_flags=list(spec.flags),
            terrain=[
                {"kind": p.kind,
                 "center": [round(float(v), 4) for v in p.center],
                 "half": [round(float(v), 4) for v in p.half],
                 "yaw": round(float(p.yaw), 4)}
                for p in spec.plateaus],
            extra={
                "box_width": grasp.box_width,
                "raw_carry_gap": grasp.raw_carry_gap,
                "has_table": bool(spec.has_table),
                "table_top_z": (float(spec.table_center[2] + spec.table_half[2])
                                if spec.has_table else 0.0),
                "box_placement": spec.placement,
                **terrain_stats,
            },
        )
        return Built(scene_xml, model, qpos, tspec, contact=contact,
                     contact_pos=contact_pos, graph=graph)

    def describe(self, info: Dict) -> str:
        """Everything worth seeing in a log or a GUI beyond the window."""
        line = f" terrain {len(info['terrain_geoms'])} geoms"
        place = info.get("box_placement") or {}
        shift = place.get("shift", [0.0, 0.0])
        if any(abs(s) > 1e-6 for s in shift) or abs(place.get("dyaw", 0.0)) > 1e-6:
            line += (f" box-shift ({shift[0]:+.2f},{shift[1]:+.2f}m,"
                     f"{np.degrees(place['dyaw']):+.0f}deg,"
                     f" pen {place['max_pen_before'] * 100:.1f}cm cleared)")
        pen = info.get("spawn_penetration") or {}
        if pen:
            line += f" SPAWN-PEN {max(pen.values()) * 1000:.0f}mm ({len(pen)} pairs)"
        return line


TASK = BoxCarryTask()
