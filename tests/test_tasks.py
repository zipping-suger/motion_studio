"""The task registry and the under-table scene reconstruction."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from studio import config, tasks
from studio.recon import assets
from studio.tasks import under_table_params as utp

# ------------------------------------------------------------- registry --


def test_registry_lists_all_tasks_with_box_carry_default():
    assert set(tasks.names()) == {"box_carry", "under_table", "kick"}
    assert tasks.DEFAULT == "box_carry"


def test_unknown_task_fails_loudly():
    with pytest.raises(SystemExit):
        tasks.load("cartwheel")


def test_box_carry_wraps_the_existing_defaults():
    """The wrapper must share config's dicts, or config.yml's top-level
    scene:/solve: sections would silently stop applying."""
    task = tasks.load("box_carry")
    assert task.scene_defaults is config.SCENE_DEFAULTS
    assert task.solve_defaults is config.SOLVE_DEFAULTS
    assert task.subtree == config.TASK_SUBTREE
    assert task.passed("LIFT") and not task.passed("DR")


def test_under_table_task_shape():
    task = tasks.load("under_table")
    assert task.subtree == config.PROCESSED_ROOT / "humanoid"
    assert task.scene_defaults is utp.SCENE_DEFAULTS
    assert "scene_seed" in task.scene_defaults
    assert task.passed("PASS") and not task.passed("failed")


def test_task_dirs_spans_all_embodiment_subtrees(tmp_path):
    root = tmp_path / "run/outputs" / config.PROCESSED_ROOT
    for sub in ("humanoid_object/box_00", "humanoid/duck_00"):
        (root / sub).mkdir(parents=True)
    assert [d.name for d in config.task_dirs(tmp_path / "run")] == [
        "box_00", "duck_00"]


def test_coerce_follows_the_defaults_type():
    assert utp.coerce("true", False) is True
    assert utp.coerce("0", True) is False
    assert utp.coerce("512", 2048) == 512
    assert utp.coerce("0.5", 1.0) == 0.5


def test_under_table_verdict_logic():
    task = tasks.load("under_table")
    ok = {"passed": True}
    bad = {"passed": False}
    assert task.verdict([]) == "error"
    assert task.verdict([("t", None)]) == "error"
    assert task.verdict([("t", bad)]) == "failed"
    assert task.verdict([("t", bad), ("u", ok)]) == "PASS"


def test_under_table_rejects_unknown_params(tmp_path):
    task = tasks.load("under_table")
    with pytest.raises(SystemExit):
        task.reconstruct(tmp_path / "x.npz", tmp_path, "t",
                         {"scene_params": {"box_mass": 1.0}})


# ------------------------------------------------- scene reconstruction --

pytestmark_assets = pytest.mark.skipif(
    not assets.available(),
    reason="G1 assets not installed (uv run studio setup --assets)")


@pytest.fixture(scope="module")
def table_mod():
    from studio.recon import table
    return table


@pytest.fixture(scope="module")
def robot(table_mod):
    return table_mod.robot_model()


def _duck_qpos(stand_z=0.793, duck_z=0.40, T=40):
    """Synthetic clip: stand, duck (pelvis dropped), stand again. Kinematic
    only — enough for the head-driven estimation, no physics plausibility."""
    qpos = np.zeros((T, 36))
    qpos[:, 3] = 1.0  # identity quat
    qpos[:, 2] = stand_z
    qpos[12:34, 2] = duck_z
    qpos[:, 0] = np.linspace(0.0, 0.3, T)  # slight forward drift
    return qpos


@pytestmark_assets
def test_robot_scene_is_robot_only_with_full_collision_set(table_mod, robot):
    import mujoco
    assert robot.nq == 36 and robot.nv == 35
    names = {mujoco.mj_id2name(robot, mujoco.mjtObj.mjOBJ_GEOM, g)
             for g in range(robot.ngeom)}
    assert "largebox_geom" not in names
    missing = set(table_mod.ROBOT_COLLISION_GEOMS) - names
    assert missing == set()


@pytestmark_assets
def test_estimate_table_invariants(table_mod, robot):
    tcfg = SimpleNamespace(**utp.SCENE_DEFAULTS)
    spec = table_mod.estimate_table(
        _duck_qpos(), robot, tcfg,
        rng=np.random.RandomState(0), verbose=False)

    # a real duck: the slab sits well below the standing head
    assert spec.meta["h_under"] <= spec.meta["H_stand"] - tcfg.min_duck + 1e-6
    # the size-quantum snap produced grid sizes
    depth = 2 * spec.slab_half[0]
    width = 2 * spec.slab_half[1]
    assert abs(depth / tcfg.size_quantum - round(depth / tcfg.size_quantum)) < 1e-6
    assert abs(width / tcfg.size_quantum - round(width / tcfg.size_quantum)) < 1e-6
    # slab + 4 rigid legs, legs spanning floor -> underside
    assert len(spec.leg_centers) == 4
    assert np.allclose(spec.leg_centers[:, 2] * 2,
                       spec.underside_z, atol=1e-9)
    # near-edge scan bound: reference penetration never exceeds the allowed
    # conflict (closeness knob) plus the deliberate drop (height knob)
    if not spec.meta["warnings"]:
        assert spec.meta["ref_min_clearance"] >= -(
            tcfg.max_ref_table_pen + tcfg.underside_drop) - 1e-6


@pytestmark_assets
def test_arm_conflict_keeps_the_table_at_the_head(table_mod, robot):
    """A reach at slab height must not push the table away from the head:
    with arm_conflict (default) the underside stays on the ducking head even
    when an arm is raised above it; whole-body placement (arm_conflict off)
    parks the slab above the raised wrist instead — the regression this
    param exists to prevent."""
    # deep duck: keeps the head-driven underside clear of the min_duck
    # ceiling, so the two placements can actually differ
    qpos = _duck_qpos(duck_z=0.25)
    qpos[12:34, 7 + 15] = -2.4  # left_shoulder_pitch: arm overhead in the duck
    on = dict(utp.SCENE_DEFAULTS, yaw_range=0.0)
    off = dict(on, arm_conflict=False)
    spec_on = table_mod.estimate_table(
        qpos, robot, SimpleNamespace(**on),
        rng=np.random.RandomState(0), verbose=False)
    spec_off = table_mod.estimate_table(
        qpos, robot, SimpleNamespace(**off),
        rng=np.random.RandomState(0), verbose=False)
    assert spec_on.underside_z < spec_off.underside_z - 0.05
    # the head actually meets the slab: the metric the task is about
    assert spec_on.meta["head_conflict"] > spec_off.meta["head_conflict"]
    assert "head_conflict" in spec_on.meta


@pytestmark_assets
def test_estimation_is_seeded(table_mod, robot):
    tcfg = SimpleNamespace(**utp.SCENE_DEFAULTS)
    q = _duck_qpos()
    a = table_mod.estimate_table(q, robot, tcfg,
                                 rng=np.random.RandomState(3), verbose=False)
    b = table_mod.estimate_table(q, robot, tcfg,
                                 rng=np.random.RandomState(3), verbose=False)
    c = table_mod.estimate_table(q, robot, tcfg,
                                 rng=np.random.RandomState(4), verbose=False)
    assert np.allclose(a.slab_center, b.slab_center) and a.yaw == b.yaw
    assert not (np.allclose(a.slab_center, c.slab_center) and a.yaw == c.yaw)


# ------------------------------------------------------------ kick task --


def _kick_qpos(T=45):
    """Synthetic clip: stand, swing the right leg forward via hip pitch,
    return. Kinematic only — enough for kick detection + placement."""
    qpos = np.zeros((T, 36))
    qpos[:, 3] = 1.0
    qpos[:, 2] = 0.793
    # right_hip_pitch is DOF 6 (see loader.DOF_NAMES): 0 -> -1.3 -> 0
    swing = np.concatenate([np.zeros(15),
                            -1.3 * np.sin(np.linspace(0, np.pi, 24)),
                            np.zeros(T - 39)])
    qpos[:, 7 + 6] = swing
    return qpos


@pytest.fixture(scope="module")
def kick_mod():
    from studio.recon import kick
    return kick


@pytestmark_assets
def test_detect_kick_finds_the_swinging_foot(kick_mod, robot):
    feet = kick_mod.foot_site_trajectory(_kick_qpos(), robot)
    side, (s0, s1), kick_frame = kick_mod.detect_kick(feet, 0.10)
    assert side == 1  # right
    assert 15 <= s0 < kick_frame <= s1 < 40


@pytestmark_assets
def test_estimate_kick_places_a_conflicting_but_kickable_box(kick_mod, robot):
    from studio.tasks import kick_params as kp
    kcfg = SimpleNamespace(**kp.SCENE_DEFAULTS)
    spec = kick_mod.estimate_kick(_kick_qpos(), robot, kcfg,
                                  rng=np.random.RandomState(0), verbose=False)
    m = spec.meta
    # the reference swing penetrates the box (the augmentation signal) ...
    assert m["ref_min_clearance"] < 0
    assert m["foot_conflict"] >= kcfg.max_ref_pen - 1e-6
    # ... but the kick point itself stays reachable and the rest of the
    # body clears, unless the estimator said otherwise
    if not m["warnings"]:
        assert m["kick_clearance"] >= kcfg.kick_margin - 1e-6
        assert m["body_clearance"] >= kcfg.body_clearance - 1e-6
    # box stands on the floor
    assert np.isclose(spec.center[2], spec.half[2])

    # seeded: same seed reproduces, another seed moves the box
    again = kick_mod.estimate_kick(_kick_qpos(), robot, kcfg,
                                   rng=np.random.RandomState(0),
                                   verbose=False)
    other = kick_mod.estimate_kick(_kick_qpos(), robot, kcfg,
                                   rng=np.random.RandomState(3),
                                   verbose=False)
    assert np.allclose(spec.center, again.center) and spec.yaw == again.yaw
    assert not (np.allclose(spec.center, other.center)
                and spec.yaw == other.yaw)


@pytestmark_assets
def test_kick_emit_writes_a_solvable_scene(kick_mod, table_mod, tmp_path):
    import mujoco
    from studio.tasks import kick_params as kp
    qpos = _kick_qpos()
    spec = kick_mod.estimate_kick(
        qpos, kick_mod.robot_model(), SimpleNamespace(**kp.SCENE_DEFAULTS),
        rng=np.random.RandomState(0), verbose=False)
    trial = table_mod.emit_obstacle_trial(
        qpos, {"fps": 30.0, "file_path": "synthetic"}, spec.boxes(),
        spec.primitives(), spec.geom_names(), tmp_path, "kick_00",
        kp.SCENE_DEFAULTS, "kick",
        {"kick_side": spec.meta["side"],
         "kick_frame": spec.meta["kick_frame"], "warnings": []})
    task_dir = trial.parent
    model = mujoco.MjModel.from_xml_path(str(task_dir / "scene.xml"))
    assert model.nq == 36
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                             "kick_obstacle_geom") >= 0
    info = json.loads((task_dir / "task_info.json").read_text())
    assert info["task_type"] == "kick"
    assert info["kick_side"] == "right"
    assert len(info["primitives"]) == 1


@pytestmark_assets
def test_emit_trial_writes_a_solvable_scene(table_mod, tmp_path):
    import mujoco
    params = dict(utp.SCENE_DEFAULTS)
    qpos = _duck_qpos()
    spec = table_mod.estimate_table(
        qpos, table_mod.robot_model(), SimpleNamespace(**params),
        rng=np.random.RandomState(0), verbose=False)
    meta = {"fps": 30.0, "file_path": "synthetic"}
    trial = table_mod.emit_trial(qpos, meta, spec, tmp_path, "duck_00", params)

    task_dir = trial.parent
    model = mujoco.MjModel.from_xml_path(str(task_dir / "scene.xml"))
    assert model.nq == 36
    for name in spec.geom_names():
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
    # every robot collision geom got its pair against slab + 4 legs
    assert model.npair >= 5 * len(table_mod.ROBOT_COLLISION_GEOMS)

    ref = np.load(trial / "trajectory_kinematic.npz")
    assert ref["qpos"].shape == (len(qpos), 36)
    assert ref["qvel"].shape == (len(qpos), 35)
    assert ref["ctrl"].shape == (len(qpos), 29)

    info = json.loads((task_dir / "task_info.json").read_text())
    assert info["task_type"] == "under_table"
    assert len(info["primitives"]) == 5
    assert info["robot_collision_geoms"] == table_mod.ROBOT_COLLISION_GEOMS
    assert info["scene_params"]["scene_seed"] == 0
