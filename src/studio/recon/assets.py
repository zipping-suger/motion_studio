"""Where the G1 assets live, and how the repo gets them.

The MuJoCo scene template is checked in — it is text, and it is the thing
we actually edit. The G1 meshes and URDF are not: ~21 MB of binary under
Unitree's own license, which this repo copies out of an installed SPIDER
at setup time rather than redistributing.

Provenance: `scene_template.xml` is the omomo `move_largebox` scene from
SPIDER's example datasets (Meta Platforms, CC-BY-NC). The meshes and
URDF are Unitree G1 assets shipped in
SPIDER's `assets/robots/unitree_g1/`.
"""

import re
import shutil
from pathlib import Path

from ..config import REPO_ROOT

ASSETS_DIR = REPO_ROOT / "assets"
TEMPLATE_SCENE = ASSETS_DIR / "scene_template.xml"

G1_DIR = ASSETS_DIR / "g1"
MESHES_DIR = G1_DIR / "meshes"
G1_URDF = G1_DIR / "g1_29dof.urdf"

# what install() pulls out of a SPIDER checkout or installed package
_SPIDER_G1 = "assets/robots/unitree_g1"
_URDF_NAME = "g1_custom_collision_29dof.urdf"

MISSING_MSG = (
    f"G1 assets are missing from {G1_DIR}\n"
    "  uv run studio setup --assets    # copies them from an installed SPIDER"
)


def available() -> bool:
    return MESHES_DIR.is_dir() and any(MESHES_DIR.glob("*.STL"))


def require() -> None:
    """Fail with instructions rather than a MuJoCo mesh-not-found error."""
    if not available():
        raise SystemExit(MISSING_MSG)


def robot_xml() -> str:
    """The scene template with `meshdir` pointed at the installed meshes.

    This doubles as the G1 robot model: the template carries the full
    29-DoF body tree inline, and its rest rotations are identical to
    SPIDER's standalone `scene.xml` (verified bit-for-bit), so nothing
    needs that second file.
    """
    require()
    xml = TEMPLATE_SCENE.read_text()
    return re.sub(r'meshdir="[^"]*"', f'meshdir="{MESHES_DIR}"', xml)


def _spider_g1_dir(spider_root: Path) -> Path:
    """Accept either a SPIDER checkout/package root or the G1 dir itself."""
    for cand in (spider_root / _SPIDER_G1,
                 spider_root / "spider" / _SPIDER_G1,
                 spider_root):
        if (cand / "meshes").is_dir():
            return cand
    raise SystemExit(f"no {_SPIDER_G1}/meshes under {spider_root}")


def required_meshes() -> set[str]:
    """Mesh files the template actually references (35 of SPIDER's 64)."""
    names = set(re.findall(r'file="([^"]+)"', TEMPLATE_SCENE.read_text()))
    # panel_app draws the rubber-hand collision hulls, which the template
    # only names when hand_geom=mesh
    return names | {"left_rubber_hand.STL", "right_rubber_hand.STL"}


def install(spider_root: Path) -> int:
    """Copy the meshes + URDF this repo needs out of a SPIDER install.
    Returns the number of files copied."""
    src = _spider_g1_dir(Path(spider_root).expanduser())
    MESHES_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = []
    for name in sorted(required_meshes()):
        s = src / "meshes" / name
        if not s.is_file():
            missing.append(name)
            continue
        shutil.copy2(s, MESHES_DIR / name)
        copied += 1
    if missing:
        raise SystemExit(f"{src / 'meshes'} is missing {len(missing)} "
                         f"mesh(es), e.g. {missing[:3]}")

    urdf = src / _URDF_NAME
    if urdf.is_file():
        shutil.copy2(urdf, G1_URDF)
        copied += 1
    license_file = src / "LICENSE"
    if license_file.is_file():
        shutil.copy2(license_file, G1_DIR / "LICENSE")
        copied += 1
    return copied
