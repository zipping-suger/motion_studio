"""Scene reconstruction: the pure-geometry parts, on synthetic clips.

These build a motion by construction (a two-hand box carry) rather than
loading a real NPZ, so they assert on behaviour the detector is supposed
to have, independent of any recorded clip.
"""

import numpy as np
import pytest

from studio.recon import grasp as G
from studio.recon.loader import KIM_LEFT_HAND_TIP, KIM_RIGHT_HAND_TIP
from studio.recon.scene import _mask_to_rects, optimize_box_placement
from studio.recon.signal import smooth, smooth3

N_KP = 34


def carry_motion(T=120, pick=40, release=90, gap=0.40, lift=0.30):
    """A clip that reaches, grips at `pick`, lifts, carries, and releases.

    Hands start apart and low, close to `gap` at the pick, rise by `lift`
    while translating, then separate again after `release`.
    """
    jp = np.zeros((T, N_KP, 3))
    for t in range(T):
        if t < pick:                       # approach: wide, descending
            g = gap + 0.35 * (1 - t / max(pick, 1))
            z = 1.0 - 0.3 * (t / max(pick, 1))
            x = 0.0
        elif t <= release:                 # carry: fixed gap, rising + moving
            f = (t - pick) / max(release - pick, 1)
            g, z, x = gap, 0.7 + lift * f, 0.6 * f
        else:                              # release: hands open, box left
            f = (t - release) / max(T - 1 - release, 1)
            g, z, x = gap + 0.5 * f, 0.7 + lift, 0.6
        jp[t, KIM_LEFT_HAND_TIP] = [x, +g / 2, z]
        jp[t, KIM_RIGHT_HAND_TIP] = [x, -g / 2, z]
    meta = {"joint_positions": jp, "n_frames": T, "fps": 30.0,
            "foot_contacts": np.ones((T, 4), dtype=bool)}
    qpos = np.zeros((T, 36))
    qpos[:, 2] = 0.8                        # pelvis height, unused by these
    return meta, qpos


# ------------------------------------------------------------ signal --

def test_smooth_preserves_length_and_flat_signals():
    x = np.full(50, 3.0)
    out = smooth(x, 7)
    assert out.shape == x.shape
    assert np.allclose(out, 3.0)


def test_smooth_attenuates_a_single_spike():
    x = np.zeros(50)
    x[25] = 1.0
    out = smooth(x, 5)
    assert out[25] == pytest.approx(0.2)
    assert out.sum() == pytest.approx(1.0)     # box filter conserves mass


def test_smooth3_is_per_axis():
    p = np.zeros((40, 3))
    p[:, 0] = np.arange(40)
    out = smooth3(p)
    assert out.shape == p.shape
    assert np.allclose(out[:, 1:], 0.0)


# ------------------------------------------------------------- grasp --

def test_detects_the_carry_window():
    meta, qpos = carry_motion(pick=40, release=90)
    g = G.detect_grasp(meta, qpos)
    # the engage trim can land a few frames either side of the true pick
    assert abs(g.pick_frame - 40) <= 8
    assert g.release_frame >= 85
    assert g.lift_height > 0.15


def test_box_width_is_the_gap_minus_two_palm_offsets():
    meta, qpos = carry_motion(gap=0.30)
    g = G.detect_grasp(meta, qpos)
    assert g.box_width == pytest.approx(0.30 - 2 * G.HAND_SURFACE_OFFSET,
                                        abs=0.02)


def test_wide_gap_clamps_the_width_and_flags_it():
    meta, qpos = carry_motion(gap=0.80)
    g = G.detect_grasp(meta, qpos)
    assert g.box_width == G.BOX_WIDTH_MAX
    assert "width_clamped" in g.quality_flags


def test_override_window_remeasures_everything_from_it():
    meta, qpos = carry_motion(pick=40, release=90, gap=0.30)
    g = G.detect_grasp(meta, qpos)
    forced = G.override_window(meta, g, 50, 80)

    assert (forced.pick_frame, forced.release_frame) == (50, 80)
    assert "window_override" in forced.quality_flags
    # width is re-derived from the FORCED window's median gap, not kept
    assert forced.box_width == pytest.approx(0.30 - 2 * G.HAND_SURFACE_OFFSET,
                                             abs=0.02)
    # detector-history flags are dropped: the window is authoritative now
    assert "stationarity_fallback" not in forced.quality_flags


def test_override_window_at_frame_zero_means_held_start():
    meta, qpos = carry_motion()
    g = G.detect_grasp(meta, qpos)
    forced = G.override_window(meta, g, 0, 100)
    assert forced.starts_held is True
    assert "starts_held" in forced.quality_flags


def test_override_window_clamps_to_the_clip():
    meta, qpos = carry_motion(T=120)
    g = G.detect_grasp(meta, qpos)
    forced = G.override_window(meta, g, -50, 9999)
    assert forced.pick_frame == 0
    assert forced.release_frame == 119


def test_override_window_forces_release_after_pick():
    meta, qpos = carry_motion(T=120)
    g = G.detect_grasp(meta, qpos)
    forced = G.override_window(meta, g, 60, 10)   # inverted on purpose
    assert forced.release_frame > forced.pick_frame


# --------------------------------------------------------- placement --

def test_placement_leaves_a_clear_rest_pose_alone():
    """Nothing sweeps the box column, so the nominal placement stands."""
    jp = np.full((40, N_KP, 3), 5.0)          # robot far away
    dxy, dyaw, stats = optimize_box_placement(
        jp, pick=40, rest_pos=np.array([0.0, 0.0, 0.8]), rest_yaw=0.0,
        half_w=0.2, half_d=0.12, top_z=0.9)
    assert np.allclose(dxy, 0.0) and dyaw == 0.0
    assert stats["max_pen_before"] == 0.0


def test_placement_shifts_the_box_out_of_a_swept_column():
    """A leg keypoint parked in the box's rest volume must move it."""
    jp = np.zeros((40, N_KP, 3))
    jp[:, :, 2] = 5.0                          # everything else out of band
    jp[:, 3] = [0.0, 0.0, 0.8]                 # a leg keypoint, in the box
    rest = np.array([0.0, 0.0, 0.8])
    dxy, dyaw, stats = optimize_box_placement(
        jp, pick=40, rest_pos=rest, rest_yaw=0.0,
        half_w=0.2, half_d=0.12, top_z=0.9)

    assert stats["max_pen_before"] > 0.0
    assert np.linalg.norm(dxy) > 0.0           # it actually moved
    assert stats["max_pen_after"] < stats["max_pen_before"]


def test_placement_is_a_no_op_when_the_clip_starts_at_the_pick():
    """pick=0 means no pre-pick frames to clear."""
    jp = np.zeros((10, N_KP, 3))
    dxy, dyaw, stats = optimize_box_placement(
        jp, pick=0, rest_pos=np.zeros(3), rest_yaw=0.0,
        half_w=0.2, half_d=0.1, top_z=0.5)
    assert np.allclose(dxy, 0.0) and dyaw == 0.0
    assert stats["checked_pts"] == 0


# ------------------------------------------------------------ carving --

def test_mask_to_rects_covers_a_full_grid_with_one_rect():
    rects = _mask_to_rects(np.ones((4, 3), dtype=bool))
    assert rects == [(0, 0, 3, 2)]


def test_mask_to_rects_covers_exactly_the_true_cells():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0:2, 0:2] = True
    mask[3, 3] = True

    covered = np.zeros_like(mask)
    for i0, j0, i1, j1 in _mask_to_rects(mask):
        assert not covered[i0:i1 + 1, j0:j1 + 1].any()   # no overlap
        covered[i0:i1 + 1, j0:j1 + 1] = True
    assert np.array_equal(covered, mask)


def test_mask_to_rects_of_an_empty_grid_is_empty():
    assert _mask_to_rects(np.zeros((3, 3), dtype=bool)) == []
