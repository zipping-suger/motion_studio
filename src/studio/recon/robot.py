"""The robot-only G1 model: the template minus the box, full body
collision set.

Every recon that needs FK of the bare robot (pick, pole, chair) builds
its model here; the trial scenes those tasks emit are built separately
(`scene.py`), this model only measures.

numpy + mujoco only (studio's own venv).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, Optional

import mujoco

from . import assets, layout

# Whole-body collision geoms activated on top of the template, which ships
# most of them commented out (thigh/wrist are active there as ghost
# self-pair partners and get OVERRIDDEN below). Torso/leg/head specs are
# the template's own; wrist/hand are enlarged to match the visual mesh
# extent.  name -> (parent body, type, size, placement)
BODY_COLLISION_GEOMS: Dict[str, tuple] = {
    "pelvis_collision": ("pelvis", "sphere", "0.07", {"pos": "0 0 -0.08"}),
    "left_hip_collision": ("left_hip_roll_link", "capsule", "0.06",
                           {"fromto": "0.02 0 0 0.02 0 -0.08"}),
    "left_thigh_collision": ("left_hip_yaw_link", "capsule", "0.055",
                             {"fromto": "-0.0 0 -0.03 -0.06 0 -0.17"}),
    "left_shin_collision": ("left_knee_link", "capsule", "0.045",
                            {"fromto": "0.01 0 0 0.01 0 -0.15"}),
    "left_linkage_brace_collision": ("left_knee_link", "capsule", "0.03",
                                     {"fromto": "0.01 0 -0.2 0.01 0 -0.28"}),
    "right_hip_collision": ("right_hip_roll_link", "capsule", "0.06",
                            {"fromto": "0.02 0 0 0.02 0 -0.08"}),
    "right_thigh_collision": ("right_hip_yaw_link", "capsule", "0.055",
                              {"fromto": "-0.0 0 -0.03 -0.06 0 -0.17"}),
    "right_shin_collision": ("right_knee_link", "capsule", "0.045",
                             {"fromto": "0.01 0 0 0.01 0 -0.15"}),
    "right_linkage_brace_collision": ("right_knee_link", "capsule", "0.03",
                                      {"fromto": "0.01 0 -0.2 0.01 0 -0.28"}),
    "torso_collision": ("torso_link", "capsule", "0.09",
                        {"fromto": "0.01 0 0.08 0.01 0 0.2"}),
    "head_collision": ("torso_link", "sphere", "0.06", {"pos": "0 0 .43"}),
    "left_shoulder_yaw_collision": ("left_shoulder_yaw_link", "capsule",
                                    "0.035", {"fromto": "0 0 -0.08 0 0 0.05"}),
    "left_elbow_yaw_collision": ("left_elbow_link", "capsule", "0.035",
                                 {"fromto": "-0.01 0 -0.01 0.08 0 -0.01"}),
    "left_forearm_collision": ("left_elbow_link", "capsule", "0.05",
                               {"fromto": "-0.01 0 -0.01 0.10 0 -0.01"}),
    "left_wrist_collision": ("left_wrist_pitch_link", "capsule", "0.05",
                             {"fromto": "-0.01 0 0 0.06 0 0"}),
    "left_hand_collision": ("left_wrist_yaw_link", "capsule", "0.06",
                            {"fromto": "0.02 0 0 0.14 -0.02 0"}),
    "right_shoulder_yaw_collision": ("right_shoulder_yaw_link", "capsule",
                                     "0.035", {"fromto": "0 0 -0.08 0 0 0.05"}),
    "right_elbow_yaw_collision": ("right_elbow_link", "capsule", "0.035",
                                  {"fromto": "-0.01 0 -0.01 0.08 0 -0.01"}),
    "right_forearm_collision": ("right_elbow_link", "capsule", "0.05",
                                {"fromto": "-0.01 0 -0.01 0.10 0 -0.01"}),
    "right_wrist_collision": ("right_wrist_pitch_link", "capsule", "0.05",
                              {"fromto": "-0.01 0 0 0.06 0 0"}),
    "right_hand_collision": ("right_wrist_yaw_link", "capsule", "0.06",
                             {"fromto": "0.02 0 0 0.14 0.02 0"}),
}

# BrainCo fingertips reach ~0.22 m past the wrist against the rubber
# hand's ~0.15, so stretch the hand slab proxy to cover the whole hand
BRAINCO_HAND_COLLISION = {
    "left_hand_collision": ("left_wrist_yaw_link", "capsule", "0.06",
                            {"fromto": "0.03 0 0 0.17 0 0"}),
    "right_hand_collision": ("right_wrist_yaw_link", "capsule", "0.06",
                             {"fromto": "0.03 0 0 0.17 0 0"}),
}


def _find_body(element: ET.Element, body_name: str) -> Optional[ET.Element]:
    if element.tag == "body" and element.get("name") == body_name:
        return element
    for child in element:
        found = _find_body(child, body_name)
        if found is not None:
            return found
    return None


def _find_geom(element: ET.Element, geom_name: str) -> Optional[ET.Element]:
    for geom in element.iter("geom"):
        if geom.get("name") == geom_name:
            return geom
    return None


def robot_scene_tree() -> ET.Element:
    """The template as an ET root, robot-only, full collision set.

    Strips the largebox body and every pair referencing it, then adds
    BODY_COLLISION_GEOMS. The template's contype/conaffinity 0 default
    keeps those inert until a <pair> names them, so this model FKs and
    simulates exactly like the template minus the box.
    """
    root = ET.fromstring(assets.robot_xml())
    worldbody = root.find("worldbody")
    contact = root.find("contact")

    box = next(b for b in worldbody.findall("body")
               if b.get("name") == "largebox")
    worldbody.remove(box)
    for pair in list(contact):
        if "largebox_geom" in (pair.get("geom1"), pair.get("geom2")):
            contact.remove(pair)

    specs = dict(BODY_COLLISION_GEOMS)
    if assets.brainco_available():
        specs.update(BRAINCO_HAND_COLLISION)
    for name, (parent_name, gtype, size, place) in specs.items():
        existing = _find_geom(worldbody, name)
        if existing is not None:
            # thigh/wrist ship ACTIVE (ghost self-pair partners): override
            # with this model's own, deliberately larger, spec
            existing.attrib.pop("pos", None)
            existing.attrib.pop("fromto", None)
            existing.set("type", gtype)
            existing.set("size", size)
            for k, v in place.items():
                existing.set(k, v)
            continue
        parent = _find_body(worldbody, parent_name)
        assert parent is not None, f"template lost body {parent_name}"
        ET.SubElement(parent, "geom", {
            "name": name, "class": "collision", "type": gtype, "size": size,
            **place,
        })
    return root


def robot_model() -> mujoco.MjModel:
    """The robot-only G1 model, for FK of a clip."""
    model = mujoco.MjModel.from_xml_string(
        ET.tostring(robot_scene_tree(), encoding="unicode"))
    layout.check_robot(model)
    return model

