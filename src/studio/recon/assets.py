"""Where the G1 assets live, and how the repo gets them.

The scene template is checked in; the meshes and URDFs are not — they are
copied out of an installed SPIDER (and, for the BrainCo hands, a local
unitree_g1 checkout) at setup time rather than redistributed. Everything
keeps working without the BrainCo variant.

Provenance: `scene_template.xml` is the omomo `move_largebox` scene from
SPIDER's example datasets (Meta Platforms, CC-BY-NC); the meshes and URDF
are Unitree G1 assets from SPIDER's `assets/robots/unitree_g1/`.
"""

import re
import shutil
from pathlib import Path

from ..config import REPO_ROOT

ASSETS_DIR = REPO_ROOT / "assets"
TEMPLATE_SCENE = ASSETS_DIR / "scene_template.xml"
# scene_template + BrainCo hands grafted on (22 finger joints, 12
# actuators), from scripts/gen_brainco_template.py; used automatically
# once the brainco assets are installed
TEMPLATE_SCENE_BRAINCO = ASSETS_DIR / "scene_template_brainco.xml"

G1_DIR = ASSETS_DIR / "g1"
MESHES_DIR = G1_DIR / "meshes"
G1_URDF = G1_DIR / "g1_29dof.urdf"

# the BrainCo-hand variant: the same 29-DoF body plus two 11-joint hands
G1_BRAINCO_URDF = G1_DIR / "g1_29dof_mode_15_brainco_hand.urdf"
BRAINCO_MESHES_DIR = G1_DIR / "meshes_brainco"

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

    Doubles as the G1 robot model: the template carries the whole 29-DoF
    body tree inline. Serves the hand-grafted template when the BrainCo
    assets are installed; the body tree is byte-identical between the two.
    """
    require()
    template = (TEMPLATE_SCENE_BRAINCO if brainco_available()
                else TEMPLATE_SCENE)
    xml = template.read_text()
    return re.sub(r'meshdir="[^"]*"', f'meshdir="{MESHES_DIR}"', xml)


def brainco_available() -> bool:
    return (G1_BRAINCO_URDF.is_file() and BRAINCO_MESHES_DIR.is_dir()
            and any(BRAINCO_MESHES_DIR.glob("*.STL")))


def install_brainco(unitree_root: Path) -> int:
    """Copy the BrainCo-hand URDF + the meshes it references out of a
    unitree_g1 checkout. Returns the number of files copied."""
    src = Path(unitree_root).expanduser()
    urdf = src / G1_BRAINCO_URDF.name
    if not urdf.is_file():
        raise SystemExit(f"no {G1_BRAINCO_URDF.name} under {src}")
    # refs are relative to the checkout root, e.g. meshes_brainco/<x>.STL
    refs = sorted(set(re.findall(r'filename="([^"]+)"', urdf.read_text())))
    missing = [r for r in refs if not (src / r).is_file()]
    if missing:
        raise SystemExit(f"{src} is missing {len(missing)} mesh(es), "
                         f"e.g. {missing[:3]}")
    copied = 0
    for r in refs:
        dst = G1_DIR / r
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / r, dst)
        copied += 1
    shutil.copy2(urdf, G1_BRAINCO_URDF)
    copied += 1
    license_file = src / "LICENSE"
    if license_file.is_file() and BRAINCO_MESHES_DIR.is_dir():
        shutil.copy2(license_file, BRAINCO_MESHES_DIR / "LICENSE")
        copied += 1
    return copied


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
    # panel_app draws the rubber-hand hulls, named only when hand_geom=mesh
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
