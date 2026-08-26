"""Template surgery: the edits every task makes to the G1 scene template
(assets.robot_xml()) to turn it into its own scene.

The template is SPIDER's omomo `move_largebox` scene — the robot, a
box named ``largebox`` on a free joint, the floor, and the hand-object
contact pairs. A task keeps the robot and the pair names and swaps the
box body for its object (`replace_object_body`), adds mesh assets and
extra pairs, restores the body collision geoms the template ships
commented out, hardens the hand-object contact and finally sizes
mujoco_warp's contact buffers to the pair count. String edits on the
XML text rather than an ElementTree, so the template's own formatting
and comments survive into the emitted scene.xml.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

# the template's insertion point for extra contact pairs
HAND_CONTACT_MARK = "<!-- hand-object contact -->"
_OBJECT_BODY = re.compile(r'<body name="largebox".*?</body>', re.S)

# the body collision geoms the template ships commented out (the box
# never meets the body), restored whenever an object can: every task
# restores the same set — thigh and wrist geoms are active in the
# template already, the old palm capsules stay off (the hands replace
# them)
BODY_COLLISION_RESTORE = (
    "pelvis_collision", "left_hip_collision", "right_hip_collision",
    "left_shin_collision", "right_shin_collision",
    "left_linkage_brace_collision", "right_linkage_brace_collision",
    "torso_collision", "head_collision",
    "left_shoulder_yaw_collision", "right_shoulder_yaw_collision",
    "left_elbow_yaw_collision", "right_elbow_yaw_collision",
)


def replace_object_body(xml: str, body_xml: str) -> str:
    """The template's largebox body becomes ``body_xml`` (keep the body
    and root-joint names, so every largebox pair still applies)."""
    xml, n = _OBJECT_BODY.subn(body_xml, xml)
    assert n == 1, "largebox body not found in template"
    return xml


def add_assets(xml: str, asset_lines: Iterable[str]) -> str:
    """Mesh (or other) asset lines appended inside <asset>."""
    lines = list(asset_lines)
    if not lines:
        return xml
    return xml.replace("</asset>",
                       "  " + "\n  ".join(lines) + "\n  </asset>")


def insert_pairs(xml: str, pair_lines: Iterable[str]) -> str:
    """Contact pair lines inserted ahead of the hand-object pairs."""
    lines = list(pair_lines)
    if not lines:
        return xml
    return xml.replace(HAND_CONTACT_MARK,
                       "\n    ".join(lines) + "\n    " + HAND_CONTACT_MARK)


def restore_body_collision(xml: str,
                           names: Sequence[str] = BODY_COLLISION_RESTORE) -> str:
    """Uncomment the named body collision geoms, typing the fromto ones
    as capsules (the collision default class is a sphere)."""
    wanted = set(names)

    def fix(m):
        kept = []
        for g in re.findall(r'<geom name="[^"]*"[^>]*?/>', m.group(1), re.S):
            name = re.search(r'name="([^"]*)"', g).group(1)
            if name not in wanted:
                continue
            if "fromto" in g and "type=" not in g:
                g = g.replace('class="collision"',
                              'class="collision" type="capsule"', 1)
            kept.append(g)
        return "\n".join(kept) if kept else m.group(0)

    xml = re.sub(r'<!--\s*((?:<geom name="[^"]*_collision"[^>]*?/>\s*)+)-->',
                 fix, xml, flags=re.S)
    for name in wanted:
        assert f'<geom name="{name}"' in xml, f"collision geom {name} not found"
    return xml


def harden_hand_object_pairs(xml: str) -> str:
    """Firmer grip and stiffer contact than the template default, so the
    squeeze force develops at millimetre- rather than centimetre-scale
    depth. Covers lh/rh plus the BrainCo template's 20 finger pairs.

    The solref timeconst must respect MuJoCo's stability bound of
    2 * timestep at the solve's 60 Hz step: below it the contact is not
    stiffer but unrealizable, and servo-squeezed fingers sank 20-37 mm
    into the object at solref 0.004.

    condim 4 adds torsional friction about the normal — capsule fingers
    touch a handle along a line, and without it a gripped pole spins
    freely. The torsional coefficient has length units, ~ mu times the
    pad radius. Sliding mu is 2.0 (rubber pad on plastic), with torsion
    carrying the twist load on top of it."""
    xml, n = re.subn(
        r'(<pair name="(?:left|right)_hand\w*_object"[^>]*)'
        r'solref="0.008 1"([^>]*)friction="1 1" condim="3"',
        r'\g<1>solref="0.04 1" solimp="0.95 0.99 0.001"\g<2>'
        r'friction="2 2 0.008" condim="4"',
        xml,
    )
    assert n in (2, 22), f"expected 2 or 22 hand-object pairs, patched {n}"
    return xml


def drop_hand_object_pairs(xml: str) -> str:
    """Remove the template's hand-object pairs (an object colliding by
    contact bitmask instead of explicit pairs)."""
    xml, n = re.subn(r'\s*<pair name="(?:left|right)_hand\w*_object"'
                     r'[^>]*?/>', "", xml)
    assert n in (2, 22), f"expected 2 or 22 hand-object pairs, dropped {n}"
    return xml


def size_contact_buffers(xml: str) -> str:
    """mujoco_warp needs collision buffers sized to the final pair count;
    a single pair can yield several contact points. Call last."""
    npair = xml.count("<pair ")
    xml = re.sub(r'<numeric data="\d+" name="max_geom_pairs" />',
                 f'<numeric data="{max(40, npair + 3)}" '
                 'name="max_geom_pairs" />', xml)
    return re.sub(r'<numeric data="\d+" name="max_contact_points" />',
                  f'<numeric data="{max(40, 2 * npair)}" '
                  'name="max_contact_points" />', xml)


def set_contact_buffers(xml: str, geom_pairs: int, contact_points: int) -> str:
    """Explicit buffer sizes, for scenes whose contacts the pair count
    cannot see (bitmask collisions)."""
    xml = re.sub(r'<numeric data="\d+" name="max_geom_pairs" />',
                 f'<numeric data="{geom_pairs}" name="max_geom_pairs" />', xml)
    return re.sub(r'<numeric data="\d+" name="max_contact_points" />',
                  f'<numeric data="{contact_points}" name="max_contact_points" />',
                  xml)
