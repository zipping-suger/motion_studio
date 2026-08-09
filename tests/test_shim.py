"""Layout shimming: the demo's save formats -> mppi_locoma's batch layout."""

import json

import numpy as np
import pytest

from studio import shim


@pytest.mark.parametrize("raw,expected", [
    ("box carrying", "box_carrying"),
    ("Box-Carrying_1", "box-carrying_1"),
    ("a person lifts a box!", "a_person_lifts_a_box"),
    ("  leading and trailing  ", "leading_and_trailing"),
    ("###", "clip"),          # nothing survives -> the fallback name
    ("", "clip"),
])
def test_sanitize(raw, expected):
    assert shim.sanitize(raw) == expected


def test_sanitize_is_idempotent():
    once = shim.sanitize("A Person Lifts A Box!")
    assert shim.sanitize(once) == once


def test_make_run_dir_suffixes_on_collision(tmp_path):
    assert shim.make_run_dir(tmp_path, "clip").name == "clip"
    assert shim.make_run_dir(tmp_path, "clip").name == "clip-2"
    assert shim.make_run_dir(tmp_path, "clip").name == "clip-3"


def _write_npz(path, contacts):
    np.savez(path, joint_positions=np.zeros((4, 3)), foot_contacts=contacts)


def test_float_foot_contacts_become_bools(tmp_path):
    """The demo writes contact probabilities; mppi_locoma's contact graph
    needs the bools batch_generate would have produced."""
    src, dst = tmp_path / "src.npz", tmp_path / "dst.npz"
    _write_npz(src, np.array([0.1, 0.5, 0.51, 0.9], dtype=np.float32))
    shim._copy_motion_npz(src, dst)

    out = np.load(dst)["foot_contacts"]
    assert out.dtype == np.bool_
    assert out.tolist() == [False, False, True, True]


def test_bool_foot_contacts_pass_through(tmp_path):
    src, dst = tmp_path / "src.npz", tmp_path / "dst.npz"
    _write_npz(src, np.array([True, False, True, True]))
    shim._copy_motion_npz(src, dst)

    out = np.load(dst)["foot_contacts"]
    assert out.dtype == np.bool_
    assert out.tolist() == [True, False, True, True]


def test_motion_without_foot_contacts_is_copied_verbatim(tmp_path):
    src, dst = tmp_path / "src.npz", tmp_path / "dst.npz"
    np.savez(src, joint_positions=np.zeros((4, 3)))
    shim._copy_motion_npz(src, dst)
    assert src.read_bytes() == dst.read_bytes()


def test_shim_example_builds_the_nested_batch_layout(tmp_path):
    """build_trial names the task <batch>_<XXXX>_<clip>, so run 'foo'
    must yield motion_0000/motion_0000_00.npz -> task foo_0000_00."""
    example = tmp_path / "example"
    example.mkdir()
    _write_npz(example / "motion.npz", np.array([True, False]))
    (example / "meta.json").write_text(json.dumps({"text": "a prompt"}))
    (example / "constraints.json").write_text("{}")

    run_dir = tmp_path / "foo"
    run_dir.mkdir()
    motion_dir = shim.shim_example(example, run_dir)

    assert motion_dir == run_dir / "motion_0000"
    assert (motion_dir / "motion_0000_00.npz").is_file()
    inputs = run_dir / "batch_inputs/motion_0000"
    assert (inputs / "meta.json").is_file()
    assert (inputs / "constraints.json").is_file()


def test_shim_example_tolerates_missing_constraints(tmp_path):
    example = tmp_path / "example"
    example.mkdir()
    _write_npz(example / "motion.npz", np.array([True]))
    (example / "meta.json").write_text("{}")

    run_dir = tmp_path / "foo"
    run_dir.mkdir()
    shim.shim_example(example, run_dir)
    assert not (run_dir / "batch_inputs/motion_0000/constraints.json").exists()


def test_shim_motion_npz_builds_the_flat_layout(tmp_path):
    """A bare Save Motion npz has no batch_inputs, which is what puts
    build_trial on its flat <dir>_<clip> naming path."""
    src = tmp_path / "box_carrying.npz"
    _write_npz(src, np.array([True, False]))

    run_dir = tmp_path / "bar"
    run_dir.mkdir()
    motion_dir = shim.shim_motion_npz(src, run_dir)

    assert motion_dir == run_dir / "bar"
    assert (motion_dir / "bar_00.npz").is_file()
    assert not (run_dir / "batch_inputs").exists()
