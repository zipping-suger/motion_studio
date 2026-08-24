"""Where DOFs live: the one module that knows both qpos layouts.

- the *body contract*: raw kimodo motion as (T, 36) — pelvis pos(3) +
  quat wxyz(4) + the 29 DOF_NAMES joints, contiguous and in order.
- the *model layout*: qpos of the compiled scene, which trial files use.
  The hand joints nest inside the wrist bodies, so the 29 body DOFs are
  NOT contiguous there — address them via `body_addr`, never by slicing.

The box free joint is always the LAST joint of a box scene, so box
addressing uses negative indices — qpos[..., -7:-4] pos, [..., -4:]
quat, [..., -5] height — valid in every layout. check_scene() enforces it.
"""

import numpy as np

BODY_NQ = 36  # the body contract: 3 pos + 4 quat + 29 joints


def _dof_names() -> list:
    from .loader import DOF_NAMES  # late: loader imports table imports us
    return DOF_NAMES


def body_addr(model) -> np.ndarray:
    """qpos address of each of the 29 body joints in ``model`` (29,)."""
    return np.array([model.joint(n).qposadr[0] for n in _dof_names()])


def to_model(model, qpos: np.ndarray) -> np.ndarray:
    """Body-contract qpos (..., 36) -> model layout (..., nq) for a
    ROBOT-ONLY model; hand joints rest at 0. Model-layout input passes
    through unchanged, so mixed callers need no case split."""
    qpos = np.asarray(qpos)
    if qpos.shape[-1] == model.nq:
        return qpos
    assert qpos.shape[-1] == BODY_NQ, (qpos.shape, model.nq)
    check_robot(model)
    out = np.zeros(qpos.shape[:-1] + (model.nq,), dtype=qpos.dtype)
    out[..., :7] = qpos[..., :7]
    out[..., body_addr(model)] = qpos[..., 7:]
    return out


def ctrl_reference(model, qpos_model: np.ndarray) -> np.ndarray:
    """Position-actuator targets (..., nu) from model-layout qpos.

    Read through the transmission, so qpos-joint order and actuator order
    are free to diverge — they already do once the hands exist."""
    import mujoco
    assert (model.actuator_trntype
            == mujoco.mjtTrn.mjTRN_JOINT).all(), "non-joint actuator"
    cols = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    return np.ascontiguousarray(qpos_model[..., cols])


def _free_joints(model) -> list:
    import mujoco
    return [j for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]


def check_robot(model) -> None:
    """Structural sanity for a robot-only model: one free joint at
    qpos[0:7], every body joint present, hinge-only elsewhere."""
    free = _free_joints(model)
    assert free == [0] and model.jnt_qposadr[0] == 0, "pelvis must be free"
    assert model.nv == model.nq - 1, (model.nq, model.nv)
    for n in _dof_names():
        model.joint(n)  # KeyError if the template lost a body joint


def check_scene(model) -> None:
    """Structural sanity for a robot+box scene: robot as above, plus the
    box free joint LAST — the invariant behind all negative indexing."""
    free = _free_joints(model)
    assert len(free) == 2 and free[0] == 0, "expected pelvis + box"
    assert free[1] == model.njnt - 1, "box joint must be last"
    assert model.jnt_qposadr[free[1]] == model.nq - 7, "box qpos must be last"
    assert model.nv == model.nq - 2, (model.nq, model.nv)
    for n in _dof_names():
        model.joint(n)
