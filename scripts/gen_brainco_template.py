"""Generate assets/scene_template_brainco.xml from the plain template.

Grafts the BrainCo hand kinematics out of the installed
`assets/g1/g1_29dof_mode_15_brainco_hand.urdf` onto both wrist bodies of
`assets/scene_template.xml`:

- 11 moving joints per hand: 6 actuated (thumb metacarpal + proximal,
  four finger proximals) and 5 distal joints coupled to their proximal
  by an <equality><joint> (the URDF's mimic joints; MJWarp implements
  equality_joint, so the coupling survives the solve).
- The tip joints are pinned in the URDF (equal limits), so they are
  baked into the tip body frame instead of becoming DOFs.
- Contact follows SPIDER's dexterous-hand scheme (see the Inspire hand
  in spider/assets/robots/inspire): a palm box plus one capsule per
  finger segment, everything else visual-only, contacts from explicit
  pairs. The primitives are fitted to the BrainCo meshes here (palm =
  mesh AABB, capsules = axis to the child joint, radius = P95 of the
  perpendicular vertex distance). The palm box keeps the template's
  `lh`/`rh` names, replacing the crude wrist spheres, so the existing
  hand-floor/hand-object pairs and plateau partners stay valid; the 20
  finger capsules get their own capsule-vs-box pairs. The rubber-hand
  visuals are replaced by the BrainCo palm + finger meshes
  (referenced as ../meshes_brainco/).
- 12 <position class="finger"> actuators append after the 29 body
  actuators, so ctrl[:29] keeps its meaning and ctrl[29:41] is hands
  (left thumb mc, thumb prox, index, middle, ring, pinky; then right).

The output is checked in; rerun this script only when the URDF or the
plain template changes:

    uv run python scripts/gen_brainco_template.py
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from studio.recon import assets  # noqa: E402

OUT = assets.ASSETS_DIR / "scene_template_brainco.xml"
MESH_PREFIX = "../meshes_brainco"  # relative to the template's meshdir

# actuated joints per hand, in ctrl order
ACTUATED = ("thumb_metacarpal", "thumb_proximal", "index_proximal",
            "middle_proximal", "ring_proximal", "pinky_proximal")
FINGER_KP = 50.0

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# finger-segment links that get a fitted collision capsule
GRASP_LINK = re.compile(  # left links say _Link, right links _link
    r"^(left|right)_(thumb|index|middle|ring|pinky)_(proximal|distal)_[Ll]ink$")


def rpy_quat(rpy):
    """URDF fixed-axis rpy -> wxyz quaternion."""
    r, p, y = rpy
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def axis_angle_quat(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)],
                           np.sin(angle / 2) * axis])


def quat_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def fmt(vals, nd=8):
    return " ".join(f"{v:.{nd}g}" for v in np.asarray(vals, float))


def stl_verts(path: Path) -> np.ndarray:
    """Triangle vertices of a binary STL, (3N, 3)."""
    raw = path.read_bytes()
    n = np.frombuffer(raw, np.uint32, count=1, offset=80)[0]
    rec = np.frombuffer(raw, np.uint8, count=n * 50, offset=84).reshape(n, 50)
    tri = rec[:, 12:48].copy().view(np.float32).reshape(n, 3, 3)
    return tri.reshape(-1, 3).astype(np.float64)


class Urdf:
    def __init__(self, path: Path):
        root = ET.parse(path).getroot()
        self.links = {l.get("name"): l for l in root.findall("link")}
        self.joints = {j.get("name"): j for j in root.findall("joint")}
        self.by_parent: dict[str, list] = {}
        for j in self.joints.values():
            parent = j.find("parent").get("link")
            self.by_parent.setdefault(parent, []).append(j)

    @staticmethod
    def origin(el):
        o = el.find("origin")
        xyz = np.fromstring(o.get("xyz", "0 0 0"), sep=" ") \
            if o is not None else np.zeros(3)
        rpy = np.fromstring(o.get("rpy", "0 0 0"), sep=" ") \
            if o is not None else np.zeros(3)
        return xyz, rpy_quat(rpy)

    def inertial_xml(self, link) -> str:
        ine = self.links[link].find("inertial")
        pos, quat = self.origin(ine)
        mass = float(ine.find("mass").get("value"))
        i = ine.find("inertia")
        ixx, iyy, izz = (float(i.get(k)) for k in ("ixx", "iyy", "izz"))
        ixy, ixz, iyz = (float(i.get(k)) for k in ("ixy", "ixz", "iyz"))
        # rotate into the body frame so quat can be omitted; MJCF
        # computes principal axes from fullinertia itself
        I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        R = quat_mat(quat)
        I = R @ I @ R.T
        full = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
        return (f'<inertial pos="{fmt(pos)}" mass="{mass:.8g}" '
                f'fullinertia="{fmt(full)}" />')

    def visual_mesh(self, link):
        vis = self.links[link].find("visual")
        if vis is None:
            return None
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            return None
        pos, quat = self.origin(vis)
        return Path(mesh.get("filename")).name, pos, quat

    def mesh_verts(self, link) -> np.ndarray:
        """Visual-mesh vertices in the link frame."""
        mesh_file, pos, quat = self.visual_mesh(link)
        verts = stl_verts(assets.BRAINCO_MESHES_DIR / mesh_file)
        return verts @ quat_mat(quat).T + pos


def collision_geom(urdf: Urdf, link: str) -> str | None:
    """Fitted collision primitive for a palm or finger-segment link.

    SPIDER's Inspire-hand contact model, fitted to the BrainCo meshes:
    the palm is its mesh AABB as a box (named lh/rh, replacing the crude
    wrist spheres), each finger segment a capsule whose axis points at
    the child joint, spanning the projected vertex extent, with radius
    the 95th-percentile perpendicular vertex distance (covers the
    knuckle bulges without inflating the contact face)."""
    side = "lh" if link.startswith("left") else "rh"
    if link.endswith("_base_link"):
        verts = urdf.mesh_verts(link)
        lo, hi = verts.min(0), verts.max(0)
        return (f'<geom name="{side}" class="collision" type="box" '
                f'pos="{fmt((lo + hi) / 2, 4)}" size="{fmt((hi - lo) / 2, 4)}" />')
    m = GRASP_LINK.match(link)
    if m is None:
        return None
    child = urdf.by_parent[link][0]  # distal or (pinned) tip joint
    cpos, _ = urdf.origin(child)
    u = cpos / np.linalg.norm(cpos)
    verts = urdf.mesh_verts(link)
    proj = verts @ u
    perp = np.linalg.norm(verts - np.outer(proj, u), axis=1)
    r = float(np.percentile(perp, 95))
    t0 = float(np.percentile(proj, 2)) + r
    t1 = float(np.percentile(proj, 98)) - r
    if m.group(3) == "distal":
        # reach through the pinned tip link: the fingertip pad belongs
        # to the distal mesh, around the tip-body origin
        t1 = max(t1, float(np.linalg.norm(cpos)))
    t1 = max(t1, t0 + 1e-4)
    name = f"{side}_{m.group(2)}{0 if m.group(3) == 'proximal' else 1}"
    return (f'<geom name="{name}" class="collision" type="capsule" '
            f'size="{r:.4f}" fromto="{fmt(t0 * u, 4)} {fmt(t1 * u, 4)}" />')


def joint_kind(j) -> str:
    """'actuated' | 'mimic' | 'pinned' | 'fixed'."""
    if j.get("type") == "fixed":
        return "fixed"
    if j.find("mimic") is not None:
        return "mimic"
    lim = j.find("limit")
    if lim is not None and lim.get("lower") == lim.get("upper"):
        return "pinned"
    return "actuated"


def emit_body(urdf: Urdf, joint, meshes: set, equalities: list,
              indent: int) -> str:
    """One URDF joint+child-link as a nested MJCF body subtree."""
    child = joint.find("child").get("link")
    pos, quat = urdf.origin(joint)
    kind = joint_kind(joint)
    if kind == "pinned":
        lim = joint.find("limit")
        angle = float(lim.get("lower"))
        if angle:  # bake the pinned angle into the body frame
            axis = np.fromstring(joint.get("axis", "1 0 0") if isinstance(
                joint.get("axis"), str) else "1 0 0", sep=" ")
            ax_el = joint.find("axis")
            if ax_el is not None:
                axis = np.fromstring(ax_el.get("xyz"), sep=" ")
            quat = quat_mul(quat, axis_angle_quat(axis, angle))

    pad = "  " * indent
    lines = [f'{pad}<body name="{child}" pos="{fmt(pos)}" '
             f'quat="{fmt(quat)}">']
    lines.append(f"{pad}  {urdf.inertial_xml(child)}")

    if kind in ("actuated", "mimic"):
        name = joint.get("name")
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        lim = joint.find("limit")
        rng = f'{lim.get("lower")} {lim.get("upper")}'
        lines.append(f'{pad}  <joint name="{name}" class="finger" '
                     f'axis="{fmt(axis)}" range="{rng}" />')
        if kind == "mimic":
            m = joint.find("mimic")
            ratio = float(m.get("multiplier", "1"))
            offset = float(m.get("offset", "0"))
            equalities.append(
                f'    <joint joint1="{name}" joint2="{m.get("joint")}" '
                f'polycoef="{offset:.8g} {ratio:.8g} 0 0 0" />')

    vis = urdf.visual_mesh(child)
    if vis is not None:
        mesh_file, vpos, vquat = vis
        mesh_name = Path(mesh_file).stem
        meshes.add(mesh_file)
        attrs = ""
        if np.any(vpos) or vquat[0] != 1.0:
            attrs = f' pos="{fmt(vpos)}" quat="{fmt(vquat)}"'
        lines.append(f'{pad}  <geom class="visual" mesh="{mesh_name}"'
                     f'{attrs} />')
        col = collision_geom(urdf, child)
        if col is not None:
            lines.append(f"{pad}  {col}")

    for j in sorted(urdf.by_parent.get(child, []),
                    key=lambda j: j.get("name")):
        lines.append(emit_body(urdf, j, meshes, equalities, indent + 1))
    lines.append(f"{pad}</body>")
    return "\n".join(lines)


def hand_subtree(urdf: Urdf, side: str, meshes: set,
                 equalities: list) -> str:
    """Both fixed children of the wrist (adapter plate + palm/fingers)."""
    parts = [emit_body(urdf, j, meshes, equalities, indent=14)
             for j in sorted(urdf.by_parent[f"{side}_wrist_yaw_link"],
                             key=lambda j: j.get("name"))]
    return "\n" + "\n".join(parts)


def main() -> None:
    urdf = Urdf(assets.G1_BRAINCO_URDF)
    template = assets.TEMPLATE_SCENE.read_text()

    meshes: set = set()
    equalities: list = []
    grafts = {side: hand_subtree(urdf, side, meshes, equalities)
              for side in ("left", "right")}

    out = template
    # graft each hand right after the wrist body's trace site
    for side, xml in grafts.items():
        anchor = f'<site name="trace_{side}_hand" size="0.01" pos="0.08 0 0" />'
        assert out.count(anchor) == 1, anchor
        out = out.replace(anchor, anchor + xml)
        # the BrainCo palm replaces the rubber-hand visual
        out = re.sub(
            r'\n *<geom pos="0.0415 0.003 0" quat="1 0 0 0" class="visual"\n'
            rf' *mesh="{side}_rubber_hand" />', "", out)

    # the fitted lh/rh palm boxes replace the crude wrist spheres
    out, n = re.subn(r'\n *<geom name="[lr]h" class="hand_collision" />',
                     "", out)
    assert n == 2, "crude lh/rh spheres not found"

    # softer servo, and armature scaled to the actual finger links: the
    # g1 default of 1.0 gives kp=50 fingers a ~0.7 s response, far too
    # slow to close before the hand lifts away
    finger_default = (
        '      <default class="finger">\n'
        f'        <position kp="{FINGER_KP:g}" dampratio="1" '
        'inheritrange="1" />\n'
        '        <joint damping="0.1" armature="0.05" frictionloss="0.0" />\n'
        '      </default>\n'
    )
    anchor = '      <default class="hip">'
    assert out.count(anchor) == 1
    out = out.replace(anchor, finger_default + anchor)

    # mesh assets (visual meshes only; collisions stay the lh/rh proxies)
    mesh_xml = "".join(
        f'    <mesh name="{Path(m).stem}" file="{MESH_PREFIX}/{m}" />\n'
        for m in sorted(meshes))
    assert out.count("  </asset>") == 1
    out = out.replace("  </asset>", mesh_xml + "  </asset>")

    # hand actuators, after the 29 body ones: ctrl[29:41]
    act_xml = "\n" + "\n".join(
        f'    <position class="finger" name="{side}_{j}_joint" '
        f'joint="{side}_{j}_joint" />'
        for side in ("left", "right") for j in ACTUATED)
    assert out.count("</actuator>") == 1
    out = out.replace("  </actuator>", act_xml + "\n  </actuator>")

    # finger-object pairs; the lh/rh palm ones already exist. Scene
    # emission upgrades them all to grip params.
    pair_xml = "\n".join(
        f'    <pair name="{side}_hand_{finger}{i}_object" '
        f'geom1="{side[0]}h_{finger}{i}" geom2="largebox_geom" '
        f'solref="0.008 1" friction="1 1" condim="3" />'
        for side in ("left", "right") for finger in FINGERS for i in (0, 1))
    marker = '\n    <pair name="largebox_floor"'
    assert out.count(marker) == 1
    out = out.replace(marker, "\n" + pair_xml + marker)

    # the mimic couplings
    eq_xml = "  <equality>\n" + "\n".join(sorted(equalities)) + \
        "\n  </equality>\n"
    out = out.replace("</mujoco>", eq_xml + "</mujoco>")

    header = ("<!-- GENERATED by scripts/gen_brainco_template.py from "
              "scene_template.xml +\n     the BrainCo-hand URDF. "
              "Edit the plain template, then rerun the script. -->\n")
    OUT.write_text(header + out)

    # compile check against the installed meshes
    import mujoco
    xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{assets.MESHES_DIR}"',
                 OUT.read_text())
    model = mujoco.MjModel.from_xml_string(xml)
    assert (model.nq, model.nv, model.nu, model.neq) == (65, 63, 41, 10), (
        model.nq, model.nv, model.nu, model.neq)
    assert model.npair == 40, model.npair  # 20 template + 20 finger pairs
    hand_geoms = [n for n in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                              for g in range(model.ngeom))
                  if n and re.match(r"^[lr]h(_|$)", n)]
    assert len(hand_geoms) == 22, hand_geoms  # 2 palms + 20 capsules
    # settle test: hold default ctrl for a second, expect no blow-up
    data = mujoco.MjData(model)
    for _ in range(60):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    print(f"wrote {OUT}: nq={model.nq} nv={model.nv} nu={model.nu} "
          f"neq={model.neq}")


if __name__ == "__main__":
    main()
