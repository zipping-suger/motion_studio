"""How studio reaches the solve venv.

The solve itself needs torch + warp + SPIDER; these cover the boundary
that studio owns — command construction and the environment handed
across it — which is pure and testable in studio's own venv.
"""

from pathlib import Path

import pytest

from studio import solve
from studio.config import REPO_ROOT, SOLVE_DEFAULTS, SOLVE_INT_KEYS, Config


def _cfg(solve_python: Path) -> Config:
    return Config(
        kimodo_repo=Path("/a/kimodo"),
        kimodo_python=Path("/a/kimodo/bin/python"),
        solve_python=solve_python,
        model="Kimodo-G1-RP-v1",
        demo_port=7860,
        encoder_host="euler",
        encoder_port=9550,
    )


@pytest.fixture
def ready(tmp_path):
    """A cfg whose solve venv exists."""
    python = tmp_path / ".venv-solve/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    return _cfg(python)


def test_missing_solve_venv_is_reported_with_the_fix(tmp_path):
    cfg = _cfg(tmp_path / "nope/bin/python")
    assert solve.available(cfg) is False
    with pytest.raises(SystemExit, match="studio setup --solve"):
        solve.require(cfg)


def test_command_targets_the_solve_venv_and_the_loop_module(ready):
    cmd = solve.command(ready, "clip_00", Path("/runs/clip/outputs"))
    assert cmd[0] == str(ready.solve_python)
    assert cmd[1:3] == ["-m", "studio.solve.loop"]
    assert "--task" in cmd and "clip_00" in cmd
    assert "/runs/clip/outputs" in cmd


def test_command_passes_params_as_repeated_flags(ready):
    cmd = solve.command(ready, "clip_00", Path("/d"),
                        {"num_samples": 512, "pos_rew_scale": 8.0})
    pairs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--param"]
    assert sorted(pairs) == ["num_samples=512", "pos_rew_scale=8.0"]


def test_command_without_params_passes_none(ready):
    assert "--param" not in solve.command(ready, "t", Path("/d"))


def test_env_puts_studio_on_the_path_and_stays_headless():
    env = solve.env()
    assert str(REPO_ROOT / "src") in env["PYTHONPATH"].split(":")
    # the demo may hold the display; the solve must never open a window
    assert env["MUJOCO_GL"] == "egl"


def test_env_preserves_an_existing_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/already/here")
    assert "/already/here" in solve.env()["PYTHONPATH"].split(":")


def test_env_has_no_trailing_separator(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert not solve.env()["PYTHONPATH"].endswith(":")


# ------------------------------------------------------- solve params --

def test_int_keys_are_a_subset_of_the_defaults():
    assert SOLVE_INT_KEYS <= set(SOLVE_DEFAULTS)


def test_the_objective_outweighs_the_pose_guide():
    """Object position is the objective; robot joints only guide. If this
    ever inverts, the solve is tracking the wrong thing."""
    assert SOLVE_DEFAULTS["pos_rew_scale"] > SOLVE_DEFAULTS["joint_rew_scale"]
