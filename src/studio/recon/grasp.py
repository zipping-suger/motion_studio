"""Grasp detection from Kimodo hand-tip kinematics (SceneBot-style hindsight).

`override_window` forces the contact window from the GUIs and re-derives
everything measured from it.

Primary signal (SceneBot, arXiv 2606.27581, Alg. 1): a contact interval is
where the key links move with LOW VELOCITY RELATIVE TO THE OBJECT. For a
two-hand box grasp whose object is itself inferred from the hands, that
becomes rigid co-movement of the hand pair — low relative velocity and a
stable gap — while the pair transports the object. Being pose-agnostic it
also covers the side and floor picks where the squat/hand-low cues fail.

Those cues (hands-low minimum, pelvis squat, root2d anchor) remain as a
fallback when no rigid interval is found, and as a cross-check flag.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .loader import FPS, KIM_LEFT_HAND_TIP, KIM_RIGHT_HAND_TIP
from .signal import smooth, smooth3

BOX_WIDTH_MIN, BOX_WIDTH_MAX = 0.06, 0.45
# the G1 hand geoms press the object at |y| ~= 0.048 in the wrist_yaw
# frame, and the Kimodo hand-tip keypoints sit at that frame's (0.1, 0, 0)
# to ~0.5 cm, so box width = keypoint gap minus two palm-face offsets
HAND_SURFACE_OFFSET = 0.048

REL_VEL_MAX = 0.35    # m/s: hand pair counts as rigid below this slip speed
GAP_STD_MAX = 0.02    # m: local (+-5 frame) gap variation of a held box
MIN_INTERVAL = 10     # frames
MIN_TRANSPORT = 0.10  # m: the pair must move the object to count as a grasp
MERGE_GAP = 5         # frames: bridge brief mask dropouts (e.g. fast reorient)
STEEP_AXIS_Z = 0.707  # |z| of the unit hand-hand axis; ~45 deg from horizontal
ENGAGE_SPEED = 0.25   # m/s: mid-hand speed at grasp onset. The object
                      # starts at rest, so contact begins at a stationary
                      # hold — this prunes walking-with-hands-forward.
HELD_START_SLACK = 3  # frames: smoothing edges may delay the mask onset


@dataclass
class GraspInfo:
    pick_frame: int
    release_frame: int          # inclusive; T-1 if carried to the end
    box_width: float            # along the hand-hand axis, m
    raw_carry_gap: float        # median carry gap before clamping
    lift_height: float          # mid-hand rise from pick to carry peak, m
    quality_flags: List[str] = field(default_factory=list)
    starts_held: bool = False   # clip begins mid-hold: no rest phase exists

    @property
    def ok(self) -> bool:
        return "no_lift" not in self.quality_flags


def hand_tracks(meta: Dict):
    """(left hand, right hand, gap) tracks in MuJoCo coords."""
    jp = meta["joint_positions"]
    lh, rh = jp[:, KIM_LEFT_HAND_TIP], jp[:, KIM_RIGHT_HAND_TIP]
    return lh, rh, np.linalg.norm(lh - rh, axis=1)


def _rigid_intervals(lh: np.ndarray, rh: np.ndarray, gap: np.ndarray) -> list:
    """Merged rigid-co-movement intervals of the hand pair (may be [])."""
    T = len(gap)
    vl = np.gradient(smooth3(lh), axis=0) * FPS
    vr = np.gradient(smooth3(rh), axis=0) * FPS
    rel_speed = np.linalg.norm(vl - vr, axis=1)
    pad = np.pad(gap, 5, mode="edge")
    local_std = np.array([pad[t : t + 11].std() for t in range(T)])
    # generous lower bound: degenerate clips carry below the physical
    # minimum (width-clamped later), and excluding them here would hide the
    # true grasp interval and let a pre-grasp phase win
    plausible = (gap > 0.09) & (
        gap < BOX_WIDTH_MAX + 2 * HAND_SURFACE_OFFSET + 0.05
    )
    mask = (rel_speed < REL_VEL_MAX) & (local_std < GAP_STD_MAX) & plausible
    if not mask.any():
        return []

    # bridge dropouts shorter than MERGE_GAP, but never across a net gap
    # change: hands CLOSING onto the box is the transition that must split
    # the pre-grasp phase from the grasp
    idx = np.where(mask)[0]
    intervals = []
    s = e = int(idx[0])
    for t in idx[1:]:
        if t - e <= MERGE_GAP and abs(gap[t] - gap[e]) < 0.05:
            e = int(t)
        else:
            intervals.append((s, e))
            s = e = int(t)
    intervals.append((s, e))
    return intervals


def _held_start_window(
    lh: np.ndarray, rh: np.ndarray, gap: np.ndarray
) -> Optional[tuple]:
    """(0, e) when the clip OPENS inside a rigid transporting interval: the
    pair is already carrying at frame 0, so there is no engage moment and
    `_find_grasp_window` would latch onto a mid-clip pause.

    Standing with stable arms is rejected by MIN_TRANSPORT, a walking
    approach by arm swing breaking rigidity at frame 0."""
    intervals = _rigid_intervals(lh, rh, gap)
    if not intervals or intervals[0][0] > HELD_START_SLACK:
        return None
    s, e = intervals[0]
    if e - s + 1 < MIN_INTERVAL:
        return None
    mid = 0.5 * (lh + rh)
    if float(np.linalg.norm(mid[: e + 1] - mid[0], axis=1).max()) \
            < MIN_TRANSPORT:
        return None
    return (0, e)


def _find_grasp_window(
    lh: np.ndarray, rh: np.ndarray, gap: np.ndarray
) -> Optional[tuple]:
    """Best rigid-co-movement interval of the hand pair, or None.

    Intervals are scored by peak transport distance, which rejects static
    pauses that never move an object; walking arm-swing is rejected by the
    relative velocity itself.
    """
    intervals = _rigid_intervals(lh, rh, gap)
    if not intervals:
        return None

    mid = 0.5 * (lh + rh)
    mid_speed = np.linalg.norm(np.gradient(smooth3(mid), axis=0), axis=1) * FPS

    def _clip_to_engage(s: int, e: int) -> Optional[tuple]:
        """Trim the interval start to the hold moment before the lift.

        The object rests until the pick, so contact begins at a stationary
        mid-hand moment. Among the stationary runs still preceding
        MIN_TRANSPORT of transport, the pick-hold is the one followed by
        the LARGEST RISE, which rejects both the initial standing pose and
        pauses after the lift. Near-ties go to the later run.
        """
        engaged = {
            int(t) for t in np.where(mid_speed[s : e + 1] < ENGAGE_SPEED)[0] + s
        }
        run_starts = [
            t for t in sorted(engaged)
            if t - 1 not in engaged
            and t <= e - MIN_INTERVAL + 1
            and float(np.linalg.norm(mid[t : e + 1] - mid[t], axis=1).max())
            >= MIN_TRANSPORT
        ]
        best_t, best_rise = None, -1.0
        for t in run_starts:  # ascending: later run wins unless clearly worse
            rise = float(mid[t : e + 1, 2].max() - mid[t, 2])
            if best_t is None or rise > best_rise - 0.05:
                best_t, best_rise = t, max(best_rise, rise)
        return (best_t, e) if best_t is not None else None

    best, best_score = None, 0.0
    for s, e in intervals:
        if e - s + 1 < MIN_INTERVAL:
            continue
        window = _clip_to_engage(s, e)
        if window is None:
            continue
        t, _ = window
        transport = float(np.linalg.norm(mid[t : e + 1] - mid[t], axis=1).max())
        # a pick LIFTS: weight rise from the engage moment so incidental
        # co-moving phases lose to the actual pick-and-lift interval
        rise = float(mid[t : e + 1, 2].max() - mid[t, 2])
        score = transport + 2.0 * max(0.0, rise)
        if score > best_score:
            best, best_score = window, score
    return best


def detect_grasp(
    meta: Dict,
    qpos: np.ndarray,
    constraints: Optional[Dict] = None,
    allow_held_start: bool = False,
) -> GraspInfo:
    """SceneBot-style detection with legacy cue-consensus fallback.

    allow_held_start accepts a rigid window beginning at frame 0 as a clip
    that STARTS holding the box. Default off: batch clips always approach
    first, so a frame-0 window there is pre-grasp co-movement."""
    lh, rh, gap = hand_tracks(meta)
    T = len(lh)
    midh = smooth(0.5 * (lh[:, 2] + rh[:, 2]))

    def _fallback() -> GraspInfo:
        legacy = _detect_grasp_legacy(meta, qpos, constraints)
        legacy.quality_flags.append("stationarity_fallback")
        return legacy

    starts_held = False
    window = _find_grasp_window(lh, rh, gap)
    if allow_held_start:
        held = _held_start_window(lh, rh, gap)
        if held is not None:
            # outranks any later window: with no engage moment the scored
            # search would misplace the rest pose at a mid-clip pause
            window, starts_held = held, True
    if window is None:
        return _fallback()

    pick, w_end = window
    # a first-frame pick is never real in approach-first data
    if pick == 0 and not starts_held:
        return _fallback()
    flags: List[str] = ["starts_held"] if starts_held else []

    # cross-check against the cue consensus; meaningless for a held start
    if not starts_held:
        cues = [int(np.argmin(midh)), int(np.argmin(smooth(qpos[:, 2])))]
        if constraints and constraints.get("pick_frame") is not None:
            cues.append(int(constraints["pick_frame"]))
        if abs(pick - int(np.median(cues))) > 20:
            flags.append("pick_cues_disagree")

    lift_thresh = midh[pick] + 0.05
    lift_height = float(midh[pick:].max() - midh[pick])
    if lift_height < 0.15:
        if starts_held:
            # a held carry may legitimately never rise (walk-and-carry)
            flags.append("no_lift")
        else:
            # if the cues CAN lift-detect, the window found a late pause
            legacy = _detect_grasp_legacy(meta, qpos, constraints)
            if "no_lift" not in legacy.quality_flags:
                legacy.quality_flags.append("stationarity_fallback")
                return legacy
            flags.append("no_lift")

    # sustained gap-opening, robust to brief rigidity breaks mid-carry
    grasp_idx = np.arange(pick, w_end + 1)
    carry_gap = float(np.median(gap[grasp_idx]))
    release = _find_release(gap, midh, pick, T, carry_gap, lift_thresh)

    # force-closure pruning cues, as quality flags rather than edge drops
    if float(np.std(gap[grasp_idx])) > 0.05:
        flags.append("unstable_gap")
    axis = (lh[grasp_idx] - rh[grasp_idx]) / gap[grasp_idx, None]
    if float(np.median(np.abs(axis[:, 2]))) > STEEP_AXIS_Z:
        # palms nearly stacked: an antipodal side grasp cannot oppose
        # gravity through friction alone
        flags.append("grasp_axis_steep")

    surface_gap = carry_gap - 2 * HAND_SURFACE_OFFSET
    box_width = float(np.clip(surface_gap, BOX_WIDTH_MIN, BOX_WIDTH_MAX))
    if abs(box_width - surface_gap) > 0.02:
        flags.append("width_clamped")

    return GraspInfo(
        pick_frame=pick,
        release_frame=release,
        box_width=box_width,
        raw_carry_gap=carry_gap,
        lift_height=lift_height,
        quality_flags=flags,
        starts_held=starts_held,
    )


def _find_release(gap, midh, pick, T, carry_gap, lift_thresh) -> int:
    """First frame where the gap opens well past the carry grip AND stays
    open for 5 frames, so one noisy frame does not cut the grasp."""
    opened = 0
    for t in range(pick, T):
        if gap[t] > carry_gap + 0.12 and midh[t] < lift_thresh:
            opened += 1
            if opened >= 5:
                return t - opened + 1
        else:
            opened = 0
    return T - 1


def _detect_grasp_legacy(
    meta: Dict,
    qpos: np.ndarray,
    constraints: Optional[Dict] = None,
) -> GraspInfo:
    lh, rh, gap = hand_tracks(meta)
    T = len(lh)
    flags: List[str] = []

    midh = smooth(0.5 * (lh[:, 2] + rh[:, 2]))
    pelvz = smooth(qpos[:, 2])

    # pick moment: consensus of hand-low, pelvis-squat, root2d anchor
    cues = [int(np.argmin(midh)), int(np.argmin(pelvz))]
    if constraints and constraints.get("pick_frame") is not None:
        cues.append(int(constraints["pick_frame"]))
    pick = int(np.median(cues))
    if max(cues) - min(cues) > 20:
        flags.append("pick_cues_disagree")
    # snap to the hand-low minimum near the consensus
    lo, hi = max(0, pick - 15), min(T, pick + 16)
    pick = lo + int(np.argmin(midh[lo:hi]))

    # carry window: hands lifted above pick height
    lift_thresh = midh[pick] + 0.05
    carry_mask = np.zeros(T, dtype=bool)
    carry_mask[pick:] = midh[pick:] > lift_thresh
    carry_idx = np.where(carry_mask)[0]
    lift_height = float(midh[pick:].max() - midh[pick])
    if len(carry_idx) < 10 or lift_height < 0.15:
        flags.append("no_lift")
        carry_idx = np.arange(pick, T)

    carry_gap = float(np.median(gap[carry_idx]))
    release = _find_release(gap, midh, pick, T, carry_gap, lift_thresh)

    # box width from the carry phase only
    if float(np.std(gap[carry_idx])) > 0.05:
        flags.append("unstable_gap")
    surface_gap = carry_gap - 2 * HAND_SURFACE_OFFSET
    box_width = float(np.clip(surface_gap, BOX_WIDTH_MIN, BOX_WIDTH_MAX))
    if abs(box_width - surface_gap) > 0.02:
        flags.append("width_clamped")

    return GraspInfo(
        pick_frame=pick,
        release_frame=release,
        box_width=box_width,
        raw_carry_gap=carry_gap,
        lift_height=lift_height,
        quality_flags=flags,
    )


def override_window(meta: Dict, grasp: GraspInfo,
                    pick: int, release: int) -> GraspInfo:
    """Force the contact window and RE-MEASURE what depends on it: box
    width from the forced window's median gap, lift from the forced pick,
    held start iff pick is frame 0. Detector-history flags are dropped;
    the window is authoritative now.
    """
    lh, rh, gap = hand_tracks(meta)
    T = len(lh)
    pick = int(np.clip(pick, 0, T - 2))
    release = int(np.clip(release, pick + 1, T - 1))

    midh = smooth(0.5 * (lh[:, 2] + rh[:, 2]))
    win = np.arange(pick, release + 1)
    carry_gap = float(np.median(gap[win]))
    surface_gap = carry_gap - 2 * HAND_SURFACE_OFFSET
    box_width = float(np.clip(surface_gap, BOX_WIDTH_MIN, BOX_WIDTH_MAX))

    flags = ["window_override"]
    if pick == 0:
        flags.append("starts_held")
    if float(np.std(gap[win])) > 0.05:
        flags.append("unstable_gap")
    if abs(box_width - surface_gap) > 0.02:
        flags.append("width_clamped")
    lift_height = float(midh[pick:].max() - midh[pick])
    if lift_height < 0.15:
        flags.append("no_lift")

    return dataclasses.replace(
        grasp, pick_frame=pick, release_frame=release, box_width=box_width,
        raw_carry_gap=carry_gap, lift_height=lift_height,
        quality_flags=flags, starts_held=(pick == 0))
