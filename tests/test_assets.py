"""G1 asset resolution and the scene template."""

import re

import pytest

from studio.recon import assets

pytestmark = pytest.mark.skipif(
    not assets.available(),
    reason="G1 assets not installed (uv run studio setup --assets)")


def test_template_is_checked_in():
    """The template is text and ours to edit, unlike the meshes."""
    assert assets.TEMPLATE_SCENE.is_file()
    assert "<mujoco" in assets.TEMPLATE_SCENE.read_text()


def test_required_meshes_covers_the_template_plus_the_hand_hulls():
    required = assets.required_meshes()
    referenced = set(re.findall(r'file="([^"]+)"',
                                assets.TEMPLATE_SCENE.read_text()))
    assert referenced <= required
    # panel_app draws these even when the template does not name them
    assert {"left_rubber_hand.STL", "right_rubber_hand.STL"} <= required


def test_every_required_mesh_was_installed():
    missing = [m for m in assets.required_meshes()
               if not (assets.MESHES_DIR / m).is_file()]
    assert missing == []


def test_robot_xml_absolutizes_meshdir():
    xml = assets.robot_xml()
    meshdir = re.search(r'meshdir="([^"]+)"', xml).group(1)
    assert meshdir == str(assets.MESHES_DIR)
    # the checked-in file stays relative, so the repo is location-independent
    assert 'meshdir="meshes"' in assets.TEMPLATE_SCENE.read_text()


def test_robot_xml_loads_as_the_full_g1_plus_box():
    """The template doubles as the robot model: 29 DoF + free-joint box."""
    import mujoco
    model = mujoco.MjModel.from_xml_string(assets.robot_xml())
    assert (model.nq, model.nv) == (43, 41)


def test_require_reports_how_to_fix_it(monkeypatch, tmp_path):
    monkeypatch.setattr(assets, "MESHES_DIR", tmp_path / "nope")
    with pytest.raises(SystemExit, match="studio setup --assets"):
        assets.require()


def test_install_rejects_a_directory_without_g1_meshes(tmp_path):
    with pytest.raises(SystemExit, match="unitree_g1"):
        assets.install(tmp_path)
