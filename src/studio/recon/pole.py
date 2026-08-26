"""Pole-object reconstruction: a clip of a person holding or carrying a
pole-like object (floor lamp, tripod, clothes stand) -> a physics trial
around the real OMOMO mesh.

The clips carry no object, so the pole is inferred from the hands. A
pole hold reads as the hand pair STACKED ALONG THE POLE — a steep unit
hand-hand axis at a plausible gap, exactly what grasp.py's box detector
rejects as `grasp_axis_steep`. One-hand carries fall back to the hand
that stays still in the PELVIS frame while the body transports it.

The object is the measured OMOMO mesh (assets/object_mesh/<name>.json,
from scripts/prep_object_mesh.py): mesh visual, handle capsule plus a
base primitive for collision. The capsule keeps the template's
``largebox_geom`` name, so the hand-object pairs, the solve and the
evaluator apply verbatim.

Object first, wrists second. The pole's axis, base track and grab
heights are estimated from the ORIGINAL motion's grasp pockets, so the
object trajectory is a property of the human motion whichever grasp
mode is chosen; only then is each holding wrist re-aimed by wrist-only
IK (pick.wrist_ik_frame) to wrap the ESTIMATED object's axis. The wrist
adapts to the object, never the object to the re-aimed wrist — the
small pocket shift that causes is what the solve's grip reward absorbs.

The trial declares one grip per holding hand — its pocket and its
anchor up the pole's axis — and an axially symmetric object, which is
all the solve and the evaluator need (recon.spec / solve.spec). `TASK`
is the ReconTask (recon.run) the pole task runs through.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from . import assets, layout, mjcf
from .grasp import hand_tracks
from .objects import (GRASPS, OBJECT_MESH_DIR, infer_by_name, load_sidecar,
                      pole_objects)
from .pick import (CLOSE_LEAD, CLOSE_RAMP, HAND_NOISE_SCALE, HOLD_PAD_S,
                   IK_ORI_FLAG, POWER_CLOSE_FRACTION, SIDE_BLEND_S,
                   WRAP_SQUEEZE, YAW_SWEEP, _closure_profile,
                   calibrate_grasp, palm_grip_track, wrist_ik_frame,
                   wrist_setup)
from .signal import smooth, smooth3
from .robot import robot_model
from .spec import (Built, Grip, Interaction, ObjectSpec, TrialSpec,
                   assemble_qpos)

# --- detection ---------------------------------------------------------
STEEP_MIN = 0.45      # |z| of the unit hand-hand axis
GAP_MIN, GAP_MAX = 0.15, 0.85   # m: plausible hand spacing along a pole
MIN_HOLD = 15         # frames: shortest credible hold
BRIDGE = 8            # frames: bridge brief mask dropouts
HELD_START_SLACK = 3  # frames: a hold this close to frame 0 starts held
ONE_HAND_STILL = 0.35  # m/s: pelvis-frame hand speed of a holding hand
MIN_TRANSPORT = 0.30  # m: pelvis OR hand travel a one-hand hold needs
GAP_STD_FLAG = 0.12   # m: hand-gap variation worth flagging
# A hip-hanging free arm passes the steep-axis mask under an extended
# holding hand and wrecks the axis. Ground truth (149 OMOMO clothesstand
# relocations): real pairs grip within 0.60 m, slanted fakes >= 0.62 m.
PAIR_GAP_MAX = 0.62   # m: median hand gap of a credible pair
PAIR_AZ_MIN = 0.80    # steep escape hatch: median |axis z| at least
PAIR_HSEP_MAX = 0.25  # m: ... with hands this close horizontally
PARK_R = 0.08         # m: a hand within this of its hold point has
                      # engaged the pole (hands engage one at a time)
PARK_LOOKBACK = 60    # frames searched before the two-hand window

# --- object placement --------------------------------------------------
BASE_CLEAR = 0.04     # m: carried-pole base clearance over the floor
AZ_MIN = 0.30         # min vertical component of the pole axis; flatter
                      # is hand-line noise, not a horizontal pole
AXIS_BLEND_S = 0.4    # s: upright rest pose -> hand-line axis after grab
POS_BLEND = 12        # frames: decay of the rest->grab position offset
PARK_BLEND_S = 0.6    # s: post-release park — axis up to vertical, a
                      # carried base set down. Those frames carry no
                      # object information and a pole left leaning or in
                      # the air falls while the reference floats.
AXIS_RATE_MAX = 1.0   # rad/s: cap on the axis swing rate; residual tilt
                      # jitter times the ~1.4 m grab lever is a 2.5 m/s
                      # base sweep only a batted pole could follow

# object="auto" reads the clip name, like ground_pick's shape keywords
OBJECT_KEYWORDS = (("lamp", "floorlamp"),
                   ("tripod", "tripod"), ("clothesstand", "clothesstand"),
                   ("stand", "clothesstand"), ("rack", "clothesstand"))


def available_objects() -> Tuple[str, ...]:
    """The pole objects whose measured sidecar exists (a handle_z band
    marks a pole; other objects sharing the dir are skipped)."""
    return pole_objects()


POLE_OBJECTS = available_objects()


# ------------------------------------------------------------- object --

@dataclass
class PoleSpec:
    """One measured pole object (see scripts/prep_object_mesh.py)."""
    name: str
    obj_file: Path              # raw OMOMO mesh, referenced in place
    mesh_scale: float           # raw -> canonical affine, split for
    mesh_quat: np.ndarray       # MuJoCo: asset scale + geom pos/quat
    mesh_pos: np.ndarray
    height: float               # m, canonical (+z handle, base at z=0)
    handle_radius: float
    handle_z: Tuple[float, float]   # collision capsule extent
    grab_z: Tuple[float, float]     # graspable band
    base: Dict                  # base primitive (cylinder disc or box)
    mass: float
    com: np.ndarray
    inertia_per_mass: np.ndarray


def infer_object(clip_name: str, object_param: str) -> str:
    if object_param != "auto":
        if object_param not in POLE_OBJECTS:
            raise SystemExit(f"unknown pole object {object_param!r} "
                             f"(use auto, {', '.join(POLE_OBJECTS)})")
        return object_param
    # a keyword only counts while its measured mesh is present
    return infer_by_name(clip_name, OBJECT_KEYWORDS, POLE_OBJECTS, "floorlamp")


def load_spec(name: str) -> PoleSpec:
    d = load_sidecar(name, "pole-object (uv run python "
                           "scripts/prep_object_mesh.py)")
    return PoleSpec(
        name=d["name"],
        obj_file=OBJECT_MESH_DIR / d["obj_file"],
        mesh_scale=float(d["mesh_scale"]),
        mesh_quat=np.array(d["mesh_quat"]),
        mesh_pos=np.array(d["mesh_pos"]),
        height=float(d["height"]),
        handle_radius=float(d["handle_radius"]),
        handle_z=tuple(d["handle_z"]),
        grab_z=tuple(d["grab_z"]),
        base=d["base"],
        mass=float(d["mass"]),
        com=np.array(d["com"]),
        inertia_per_mass=np.array(d["inertia_per_mass"]),
    )


# ---------------------------------------------------------- detection --

@dataclass
class PoleInfo:
    hands: List[str]            # holding hands, first-to-engage first
                                # (ties go to the lower hand)
    grab_frame: int             # the FIRST hand's engagement
    end_frame: int              # inclusive; T-1 when held to the end
    starts_held: bool
    transport: float            # peak pelvis displacement over the hold
    hand_grabs: List[int] = field(default_factory=list)
                                # per-hand engagement, aligned with
                                # `hands`; a second hand JOINS later
    hand_ends: List[int] = field(default_factory=list)
                                # per-hand release; all == end_frame
                                # unless apply_hand_windows set them

    quality_flags: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return "no_hold" not in self.quality_flags

    @property
    def primary(self) -> str:
        return self.hands[0]

    @property
    def ends(self) -> List[int]:
        return self.hand_ends or [self.end_frame] * len(self.hands)


def _runs(mask: np.ndarray, bridge: int) -> List[Tuple[int, int]]:
    """True-runs of a boolean mask, dropouts shorter than `bridge`
    merged."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    runs = []
    s = e = int(idx[0])
    for t in idx[1:]:
        if t - e <= bridge:
            e = int(t)
        else:
            runs.append((s, e))
            s = e = int(t)
    runs.append((s, e))
    return runs


def _pelvis_yaw(qpos: np.ndarray) -> np.ndarray:
    rq = qpos[:, 3:7]
    return np.unwrap(np.arctan2(
        2 * (rq[:, 0] * rq[:, 3] + rq[:, 1] * rq[:, 2]),
        1 - 2 * (rq[:, 2] ** 2 + rq[:, 3] ** 2)))


def _pelvis_frame_speed(track: np.ndarray, qpos: np.ndarray,
                        fps: float) -> np.ndarray:
    """Hand speed in the pelvis frame (m/s): a holding hand is quiet
    there while the body transports it; a free arm swings."""
    psi = _pelvis_yaw(qpos)
    c, sn = np.cos(-psi), np.sin(-psi)
    d = track - qpos[:, :3]
    local = np.stack([c * d[:, 0] - sn * d[:, 1],
                      sn * d[:, 0] + c * d[:, 1], d[:, 2]], axis=1)
    return np.linalg.norm(np.gradient(smooth3(local), axis=0),
                          axis=1) * fps


def _park_frame(track: np.ndarray, s: int) -> int:
    """Earliest frame from which the hand stays within PARK_R of its
    position at s — when it actually engaged the pole."""
    ref = track[s]
    park = s
    for t in range(s, max(0, s - PARK_LOOKBACK) - 1, -1):
        if np.linalg.norm(track[t] - ref) > PARK_R:
            break
        park = t
    return park


def _reach(track: np.ndarray, qpos: np.ndarray, s: int, e: int) -> float:
    """Median horizontal pelvis-to-hand distance over [s, e]: the
    holding hand is the extended one. Picks the on-pole hand 31/31 on
    the OMOMO clothesstand ground truth, against 20/31 for stillness."""
    return float(np.median(np.linalg.norm(
        track[s:e + 1, :2] - qpos[s:e + 1, :2], axis=1)))


def detect_pole_hold(meta: Dict, qpos: np.ndarray) -> PoleInfo:
    """The holding hand(s), each hand's own engagement, and the hold
    window — all detected, no knobs.

    Hands engage one at a time: each hand's grab is when IT parked on
    the pole, and the interaction starts at the first. A candidate pair
    must grip within PAIR_GAP_MAX (or be a steep, horizontally compact
    stack), else the hold demotes to the reaching hand alone
    (``pair_demoted``)."""
    lh, rh, gap = hand_tracks(meta)
    track = {"left": lh, "right": rh}
    T = len(lh)
    axis_z = np.abs((lh[:, 2] - rh[:, 2]) / np.maximum(gap, 1e-9))
    hold_mask = (axis_z > STEEP_MIN) & (gap > GAP_MIN) & (gap < GAP_MAX)
    runs = [r for r in _runs(hold_mask, BRIDGE) if r[1] - r[0] + 1 >= MIN_HOLD]

    flags: List[str] = []
    hands: Optional[List[str]] = None
    if runs:
        s, e = max(runs, key=lambda r: r[1] - r[0])
        gap_med = float(np.median(gap[s:e + 1]))
        az_med = float(np.median(axis_z[s:e + 1]))
        hsep_med = float(np.median(np.linalg.norm(
            (rh - lh)[s:e + 1, :2], axis=1)))
        if gap_med <= PAIR_GAP_MAX or (az_med >= PAIR_AZ_MIN
                                       and hsep_med <= PAIR_HSEP_MAX):
            # first-to-engage first; ties to the lower hand, which sets
            # the pole's floor clearance
            parks = {"left": _park_frame(lh, s),
                     "right": _park_frame(rh, s)}
            mean_z = {"left": float(lh[s:e + 1, 2].mean()),
                      "right": float(rh[s:e + 1, 2].mean())}
            hands = sorted(("left", "right"),
                           key=lambda h: (parks[h], mean_z[h]))
            grabs = [parks[hands[0]],
                     max(parks[hands[1]], parks[hands[0]])]
            s = grabs[0]        # the hold starts at the one-hand grab
            if float(gap[s:e + 1].std()) > GAP_STD_FLAG:
                flags.append("gap_unstable")
        else:
            # a hanging free arm faked the pair: same window, one hand
            hand = max(("left", "right"),
                       key=lambda h: _reach(track[h], qpos, s, e))
            hands = [hand]
            grabs = [_park_frame(track[hand], s)]
            s = grabs[0]
            flags += ["one_hand_hold", "pair_demoted"]
    else:
        # one-hand hold: the holding hand is quiet in the pelvis frame
        # while the interaction transports it, by walking or arm alone
        cand: Dict[str, Tuple[int, int]] = {}
        for hand, tr in track.items():
            speed = _pelvis_frame_speed(tr, qpos, meta["fps"])
            best = None
            for s0, e0 in _runs(speed < ONE_HAND_STILL, BRIDGE):
                if e0 - s0 + 1 < MIN_HOLD:
                    continue
                move = float(np.linalg.norm(
                    qpos[s0:e0 + 1, :2] - qpos[s0, :2], axis=1).max())
                hand_move = float(np.linalg.norm(
                    tr[s0:e0 + 1, :2] - tr[s0, :2], axis=1).max())
                if max(move, hand_move) < MIN_TRANSPORT:
                    continue
                if best is None or e0 - s0 > best[1] - best[0]:
                    best = (s0, e0)
            if best is not None:
                cand[hand] = best
        if not cand:
            return PoleInfo(["left"], 0, T - 1, False, 0.0,
                            hand_grabs=[0], hand_ends=[T - 1],
                            quality_flags=["no_hold"])
        hand = max(cand, key=lambda h: _reach(track[h], qpos, *cand[h]))
        s, e = cand[hand]
        hands = [hand]
        grabs = [s]
        flags.append("one_hand_hold")

    starts_held = s <= HELD_START_SLACK
    if starts_held:
        s = 0
        grabs = [0] * len(hands)
        flags.append("starts_held")
    if e < T - 6:
        flags.append("released_early")
    # peak, not endpoint-to-endpoint: a carry out and back still counts
    transport = float(np.linalg.norm(
        qpos[s:e + 1, :2] - qpos[s, :2], axis=1).max())
    return PoleInfo(hands, int(s), int(e), starts_held, transport,
                    hand_grabs=[int(g) for g in grabs],
                    hand_ends=[int(e)] * len(hands),
                    quality_flags=flags)


# ------------------------------------------------- user hand windows --

HAND_WINDOW_RE = re.compile(r"(\d+)-(\d+)?")


def _parse_hand_window(spec: str, hand: str,
                       T: int) -> Optional[Tuple[int, ...]]:
    """'auto' -> None (keep the detection), 'off' -> () (this hand never
    holds), 'S-E' or 'S-' -> (S, E) inclusive frames, E defaulting (and
    clamped) to the last frame."""
    if spec == "auto":
        return None
    if spec == "off":
        return ()
    m = HAND_WINDOW_RE.fullmatch(spec)
    if not m:
        raise SystemExit(f"bad {hand} hand window {spec!r} "
                         "(use auto, off, S-E or S-)")
    s = int(m.group(1))
    e = T - 1 if m.group(2) is None else min(int(m.group(2)), T - 1)
    if s > e:
        raise SystemExit(f"bad {hand} hand window {spec!r}: "
                         f"start f{s} is past end f{e}")
    return s, e


def apply_hand_windows(info: PoleInfo, meta: Dict, qpos: np.ndarray,
                       left: str = "auto",
                       right: str = "auto") -> PoleInfo:
    """User-specified per-hand contact windows over the detection: each
    hand is auto (keep the detection), off (never holds), or an explicit
    inclusive window S-E. Any override rebuilds the PoleInfo from the
    resulting windows, which also rescues a clip the detector rejected.
    The primary hand's window should span the hold — the object rides
    ITS grasp pocket."""
    T = len(qpos)
    specs = {"left": _parse_hand_window(left, "left", T),
             "right": _parse_hand_window(right, "right", T)}
    if all(v is None for v in specs.values()):
        return info
    detected = (dict(zip(info.hands, zip(info.hand_grabs, info.ends)))
                if info.ok else {})
    windows: Dict[str, Tuple[int, int]] = {}
    for hand, spec in specs.items():
        if spec is None:
            if hand in detected:
                windows[hand] = detected[hand]
        elif spec != ():
            windows[hand] = spec
    if not windows:
        return PoleInfo(["left"], 0, T - 1, False, 0.0,
                        hand_grabs=[0], hand_ends=[T - 1],
                        quality_flags=["no_hold"])
    # same ordering as the detector: first-to-engage, ties to the lower
    lh, rh, _ = hand_tracks(meta)
    track = {"left": lh, "right": rh}
    mean_z = {h: float(track[h][w[0]:w[1] + 1, 2].mean())
              for h, w in windows.items()}
    hands = sorted(windows, key=lambda h: (windows[h][0], mean_z[h]))
    grabs = [windows[h][0] for h in hands]
    ends = [windows[h][1] for h in hands]
    s, e = grabs[0], max(ends)
    starts_held = s <= HELD_START_SLACK
    if starts_held:
        grabs = [0 if g <= HELD_START_SLACK else g for g in grabs]
        s = 0
    flags = ["hand_windows"]
    if len(hands) == 1:
        flags.append("one_hand_hold")
    if starts_held:
        flags.append("starts_held")
    if e < T - 6:
        flags.append("released_early")
    transport = float(np.linalg.norm(
        qpos[s:e + 1, :2] - qpos[s, :2], axis=1).max())
    return PoleInfo(hands, int(s), int(e), starts_held, transport,
                    hand_grabs=[int(g) for g in grabs],
                    hand_ends=[int(x) for x in ends],
                    quality_flags=flags)


# --------------------------------------------------- object trajectory --

def _pole_axis(tracks: List[np.ndarray], hold: slice):
    """Unit up-the-handle axis per frame (T, 3). Two hands: the smoothed
    lower->upper line, clamped to at least AZ_MIN vertical; one hand:
    vertical. Also returns the fraction of clamped frames."""
    T = len(tracks[0])
    if len(tracks) < 2:
        return np.tile(np.array([0.0, 0.0, 1.0]), (T, 1)), 0.0
    raw = tracks[1] - tracks[0]
    if float(np.median(raw[hold, 2])) < 0:
        raw = -raw
    a = np.stack([smooth(raw[:, i], 9) for i in range(3)], axis=1)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    flat = a[:, 2] < AZ_MIN
    if flat.any():
        h = a[flat, :2]
        h = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-9)
        a[flat, :2] = h * np.sqrt(1.0 - AZ_MIN ** 2)
        a[flat, 2] = AZ_MIN
    return a, float(flat.mean())


def _blend_axis_from_vertical(a: np.ndarray, join: int,
                              fps: float) -> np.ndarray:
    """The pole hangs vertical while ONE hand holds it; once the second
    hand joins (frame `join`) the axis blends into the hand line. join
    0 (both hands hold from the start) keeps the hand line throughout."""
    if join <= 0:
        return a
    blend = max(int(AXIS_BLEND_S * fps), 1)
    wb = np.clip((np.arange(len(a)) - join) / blend, 0.0, 1.0)[:, None]
    out = (1 - wb) * np.array([0.0, 0.0, 1.0]) + wb * a
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True),
                            1e-9)


def _limit_axis_rate(a: np.ndarray, max_step: float) -> np.ndarray:
    """Unit-vector path with the per-frame swing capped at max_step
    radians: clamped forward and backward passes averaged (no lag
    bias), renormalized."""
    def clamp(path):
        out = path.copy()
        for t in range(1, len(out)):
            prev, tgt = out[t - 1], path[t]
            ang = np.arccos(float(np.clip(prev @ tgt, -1.0, 1.0)))
            if ang <= max_step:
                out[t] = tgt
            else:
                w = max_step / ang
                v = (1 - w) * prev + w * tgt
                out[t] = v / np.linalg.norm(v)
        return out
    lim = 0.5 * (clamp(a) + clamp(a[::-1])[::-1])
    return lim / np.linalg.norm(lim, axis=1, keepdims=True)


def _quat_z_axis(quats: np.ndarray) -> np.ndarray:
    """The body z axis per frame, (T, 4) wxyz -> (T, 3): the estimated
    pole axis read back off the object trajectory."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    return np.stack([2 * (x * z + w * y),
                     2 * (y * z - w * x),
                     1 - 2 * (x * x + y * y)], axis=1)


def _axis_quats(a: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Body quats (T, 4 wxyz) with the z axis along a(t) and the x axis
    parallel-transported from x0 — continuous, no roll spin."""
    T = len(a)
    quats = np.zeros((T, 4))
    x = x0 - (x0 @ a[0]) * a[0]
    n = np.linalg.norm(x)
    x = x / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    for t in range(T):
        x = x - (x @ a[t]) * a[t]
        n = np.linalg.norm(x)
        if n < 1e-6:
            x = np.cross(np.array([0.0, 0.0, 1.0]), a[t])
            n = np.linalg.norm(x)
        x /= n
        R = np.stack([x, np.cross(a[t], x), a[t]], axis=1)
        mujoco.mju_mat2Quat(quats[t], np.ascontiguousarray(R).ravel())
    return quats


def build_pole_trajectory(
    pocket_tracks: List[np.ndarray],
    info: PoleInfo,
    spec: PoleSpec,
    fps: float,
    approach_yaw: float,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray, List[str]]:
    """(obj_qpos (T, 7), {hand: grab height on the handle}, rest_pos,
    flags). The body origin is the pole's canonical base.

    Before the grab the pole stands upright under the FIRST hand's
    pocket; from that grab on the base rides grab-height below it, axis
    vertical until the second hand joins and then blending into the hand
    line. A carry that starts held hangs the pole so its base just
    clears the floor at the lowest point of the hold.

    The interaction is bracketed by a STABLE object — upright on its
    base before the grab, parked the same way after an early release.
    A reference that floats where no hand holds it cannot be tracked.
    """
    T = len(pocket_tracks[0])
    p1 = pocket_tracks[0]
    s, e = info.grab_frame, info.end_frame
    join = info.hand_grabs[1] if len(pocket_tracks) == 2 else 0
    hold = slice(s, e + 1)
    gl, gh = spec.grab_z
    flags: List[str] = []

    a, flat_frac = _pole_axis(pocket_tracks, hold)
    if flat_frac > 0.1:
        flags.append("axis_flat")
    a = _blend_axis_from_vertical(a, join, fps)
    a = _limit_axis_rate(a, AXIS_RATE_MAX / fps)
    if not info.starts_held:
        # the parked pole stands exactly upright until the first hand
        # arrives; the rate limit's backward pass would otherwise smear
        # the held tilt into the parked frames
        a = _blend_axis_from_vertical(a, s, fps)

    anchors: Dict[str, float] = {}
    if info.starts_held:
        # hang it so the base just clears the floor over the whole hold
        h1 = float(np.min((p1[hold, 2] - BASE_CLEAR) / a[hold, 2]))
    else:
        # a standing pole is gripped at the height the hand arrives
        h1 = float(p1[s, 2])
    h1c = float(np.clip(h1, gl, gh))
    if h1c > h1 + 1e-9:
        flags.append("base_scrapes")    # forced deeper on the handle
    elif h1c < h1 - 1e-9:
        flags.append("carried_high")    # hand above the handle band
    h1 = h1c
    base = p1 - h1 * a
    if float(base[hold, 2].min()) < -0.01 and "base_scrapes" not in flags:
        flags.append("base_scrapes")
    anchors[info.hands[0]] = h1
    if len(pocket_tracks) == 2:
        d = np.einsum("ti,ti->t", pocket_tracks[1] - pocket_tracks[0], a)
        # measured over the frames the second hand actually holds
        h2 = h1 + float(np.median(d[join:info.ends[1] + 1]))
        h2c = float(np.clip(h2, gl, gh))
        if abs(h2c - h2) > 0.02 and "grab_off_handle" not in flags:
            flags.append("grab_off_handle")
        anchors[info.hands[1]] = h2c

    if e < T - 1:
        # released: park standing on the base, axis blending upright.
        # The blend fits the remaining frames so the last one is always
        # fully parked — the solve rights the pole before the hands go.
        blend = max(min(int(PARK_BLEND_S * fps), T - 1 - e), 1)
        wb = np.clip((np.arange(T) - e) / blend, 0.0, 1.0)[e + 1:, None]
        base[e + 1:] = base[e]
        base[e + 1:, 2] = (1.0 - wb[:, 0]) * base[e, 2]
        av = (1 - wb) * a[e] + wb * np.array([0.0, 0.0, 1.0])
        a[e + 1:] = av / np.linalg.norm(av, axis=1, keepdims=True)

    x0 = np.array([np.cos(approach_yaw), np.sin(approach_yaw), 0.0])
    quats = _axis_quats(a, x0)

    rest = np.array([p1[s, 0], p1[s, 1], 0.0])
    off = rest - base[s]        # zero unless the grab height was clamped
    obj = np.zeros((T, 7))
    for t in range(T):
        if t < s:
            obj[t, :3] = rest
            obj[t, 3:] = quats[s]
        else:   # held, then (past e) parked standing on its base
            w = (max(0.0, 1.0 - (t - s) / POS_BLEND)
                 if not info.starts_held else 0.0)
            obj[t, :3] = base[t] + w * off
            obj[t, 3:] = quats[t]
    if e < T - 1 and obj[e, 2] > 0.08:
        flags.append("release_in_air")
    return obj, anchors, obj[0, :3].copy(), flags


# ----------------------------------------------------- wrist retarget --
POCKET_DRIFT_MAX = 0.09  # m: extra pocket-to-anchor distance the
                         # re-aimed wrist may add over the clip's own
                         # wrist. Past it the reference hand hovers off
                         # the handle and tracking fights the grip
                         # reward (carry_lamp: 0.13 m lost the 3 kg lamp
                         # mid-carry, 0.06 m carried fine). Measured
                         # against the clip-wrist baseline, so a
                         # residual offset the clip carries never counts
                         # against the retarget.
POCKET_LATERAL_MAX = 0.035  # m: how far off the pole AXIS the achieved
                            # pocket may sit and still be a wrap — past
                            # this the 13-15 mm handle is outside the
                            # closed fingers. Absolute: the clip's own
                            # wrist keeps the pocket on the axis by
                            # construction.
YAW_REFINE = (15.0, 7.5)    # deg: refinement around the sweep winner;
                            # the lateral valley is narrower than the
                            # 30-degree grid

def pole_grasp_frame(pocket: np.ndarray, axis: np.ndarray, yaw: float,
                     z_sign: float) -> np.ndarray:
    """Palm-site world orientation for a pole wrap: knuckle line (site
    z) along the pole axis thumb-up, palm normal facing the pole from
    the approach yaw. pick.side_grasp_frame for an arbitrary axis."""
    a = z_sign * axis / np.linalg.norm(axis)
    approach = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y_dir = approach - (approach @ a) * a
    n = np.linalg.norm(y_dir)
    if n < 1e-6:                # degenerate: approach parallel to axis
        y_dir = np.cross(a, np.array([0.0, 0.0, 1.0]))
        n = np.linalg.norm(y_dir)
        if n < 1e-6:
            y_dir, n = np.array([0.0, 1.0, 0.0]), 1.0
    sp = 1.0 if pocket[1] >= 0 else -1.0    # palm-normal sign per hand
    y = sp * y_dir / n
    return np.stack([np.cross(y, a), y, a], axis=1)


def retarget_pole_hand(
    qpos_robot: np.ndarray, meta: Dict, info: PoleInfo, hand: str,
    pocket: np.ndarray, obj_qpos: np.ndarray, anchor_h: float,
    grab: Optional[int] = None, end: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """Re-aim one holding hand's wrist so the fingers wrap the pole's
    instantaneous axis.

    From SIDE_BLEND_S before THIS hand's grab the wrist blends out of
    the clip's orientation into the wrap frame and holds it through this
    hand's own window; an early release blends back. Wrist-only,
    orientation-only DLS IK per frame (pick.wrist_ik_frame), best of two
    seeds, warm-started, so the clip keeps authority over the arm.
    obj_qpos is the ESTIMATED object's trajectory, never moved by the
    retarget; the wrap frames target its axis.

    Re-aiming swings the pocket around the wrist center, so among the
    yaws holding the wrap orientation (error under IK_ORI_FLAG) the
    sweep picks the one whose ACHIEVED pocket lands closest to its grip
    anchor (anchor_h up the object's axis). Scored on a frame stride;
    the winner runs the full window. Falls back to the clip's own wrist
    (mode "reference") when no yaw stays under IK_ORI_FLAG, when the
    pocket sits over POCKET_LATERAL_MAX off the axis, or when it drifts
    over POCKET_DRIFT_MAX farther off the anchor than the clip's wrist
    — orientation feasibility alone does not mean the hand still holds
    the pole where the grip reward expects it."""
    model = robot_model()
    data = mujoco.MjData(model)
    sid, qadr, vadr, jrange, cols = wrist_setup(model, hand)
    T = len(qpos_robot)
    fps = meta["fps"]
    s = info.grab_frame if grab is None else grab
    e = info.end_frame if end is None else end
    axis_t = _quat_z_axis(obj_qpos[:, 3:])
    base_t = obj_qpos[:, :3]

    # natural knuckle sense: thumb end toward the pole top (the rule
    # pick.wrap_frame_candidates uses for cylinders)
    jname = f"{hand}_thumb_proximal_joint"
    tjb = (model.jnt_bodyid[model.joint(jname).id]
           if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname) >= 0
           else -1)
    data.qpos[:] = layout.to_model(model, qpos_robot[s:s + 1])[0]
    mujoco.mj_kinematics(model, data)
    R_site = data.site_xmat[sid].reshape(3, 3)
    t_off = (R_site.T @ (data.xpos[tjb] - data.site_xpos[sid])
             if tjb >= 0 else np.zeros(3))
    z_sign = 1.0 if t_off[2] >= 0 else -1.0

    # anchor the sweep on where the clip's palm already faces at the
    # grab; fall back to the body-relative reach direction
    lh, rh, _ = hand_tracks(meta)
    tip = lh if hand == "left" else rh
    reach = tip[s, :2] - meta["joint_positions"][s, 0, :2]
    yaw0 = float(np.arctan2(reach[1], reach[0]))
    face = (1.0 if pocket[1] >= 0 else -1.0) * R_site[:, 1]
    if np.linalg.norm(face[:2]) > 1e-6:
        yaw0 = float(np.arctan2(face[1], face[0]))

    # the wrap frame yaws with the pelvis (identity at the grab): the
    # pole is axis-symmetric so every yaw is the same wrap, and a
    # world-pinned one would demand impossible wrists once the clip turns
    psi = _pelvis_yaw(qpos_robot)

    blend = max(int(SIDE_BLEND_S * fps), 1)
    t0 = max(s - blend, 0)
    w = np.zeros(T)
    w[t0:s + 1] = np.linspace(0.0, 1.0, s - t0 + 1)
    w[s:e + 1] = 1.0
    if e < T - 6:               # this hand releases early: blend back
        out_end = min(e + blend, T - 1)
        w[e:out_end + 1] = np.linspace(1.0, 0.0, out_end - e + 1)

    bases = layout.to_model(model, qpos_robot)

    # drift baseline: how far off the anchor the clip's OWN wrist keeps
    # the pocket, sampled like the cheap scoring pass (~0 by
    # construction — the pole is placed from these pockets)
    ref_dists = []
    for t in range(s, e + 1, 3):
        data.qpos[:] = bases[t]
        mujoco.mj_kinematics(model, data)
        pw = data.site_xpos[sid] + data.site_xmat[sid].reshape(3, 3) @ pocket
        ref_dists.append(np.linalg.norm(
            pw - (base_t[t] + anchor_h * axis_t[t])))
    ref_dist = float(np.mean(ref_dists)) if ref_dists else 0.0

    def frame_at(yaw_off: float, t: int) -> np.ndarray:
        return pole_grasp_frame(pocket, axis_t[t],
                                yaw0 + yaw_off + psi[t] - psi[s], z_sign)

    def solve_frame(wrist_prev, yaw_off, t):
        """Best of the chained warm start and the clip's own wrist pose
        — a single chained seed is branch-sensitive, as in pick.py.
        Returns (wrist, ori error, achieved pocket world position)."""
        R_t = frame_at(yaw_off, t)
        best = None
        for seed in (wrist_prev, qpos_robot[t, cols]):
            wrist, site, R_c, ori = wrist_ik_frame(
                model, data, sid, qadr, vadr, jrange,
                bases[t].copy(), seed.copy(), R_t)
            if best is None or ori < best[1]:
                best = (wrist, ori, site + R_c @ pocket)
        return best

    def run_pass(yaw_off: float, stride: int):
        """(mean ori error, mean pocket-to-anchor distance, mean
        pocket-to-axis lateral distance — over full-weight frames — plus
        the per-frame wrist). stride > 1 scores on a subsampled chain."""
        wrist = qpos_robot[t0, cols].copy()
        solved: Dict[int, np.ndarray] = {}
        oris, dists, lats = [], [], []
        for t in range(t0, T):
            if w[t] == 0.0 and t > t0:
                continue
            if stride > 1 and (t - t0) % stride and t != s:
                continue
            wrist, ori, pocket_w = solve_frame(wrist, yaw_off, t)
            solved[t] = wrist.copy()
            if w[t] == 1.0:
                oris.append(ori)
                dists.append(np.linalg.norm(
                    pocket_w - (base_t[t] + anchor_h * axis_t[t])))
                v = pocket_w - base_t[t]
                lats.append(np.linalg.norm(
                    v - (v @ axis_t[t]) * axis_t[t]))
        if not oris:
            return np.inf, np.inf, np.inf, solved
        return (float(np.mean(oris)), float(np.mean(dists)),
                float(np.mean(lats)), solved)

    # among orientation-feasible candidates the closest pocket wins;
    # the nearest-first sweep breaks ties toward the smallest correction
    best = None
    for yaw_off in YAW_SWEEP:
        ori, dist, _lat, _ = run_pass(yaw_off, stride=3)
        key = (ori > IK_ORI_FLAG, dist if ori <= IK_ORI_FLAG else ori)
        if best is None or key < (best[1], best[2] - 1e-6):
            best = (yaw_off, *key)
    for step in YAW_REFINE:
        for cand in (best[0] - np.radians(step), best[0] + np.radians(step)):
            ori, dist, _lat, _ = run_pass(cand, stride=3)
            key = (ori > IK_ORI_FLAG, dist if ori <= IK_ORI_FLAG else ori)
            if key < (best[1], best[2] - 1e-6):
                best = (cand, *key)
    ori, dist, lat, solved = run_pass(best[0], stride=1)
    if ori > IK_ORI_FLAG:
        return qpos_robot, {"hand": hand, "mode": "reference",
                            "fallback": "ik_ori",
                            "ik_ori_err": round(ori, 3)}
    if lat > POCKET_LATERAL_MAX:
        return qpos_robot, {"hand": hand, "mode": "reference",
                            "fallback": "pocket_off_axis",
                            "pocket_lateral_err": round(lat, 3)}
    if dist > ref_dist + POCKET_DRIFT_MAX:
        return qpos_robot, {"hand": hand, "mode": "reference",
                            "fallback": "pocket_drift",
                            "pocket_anchor_err": round(dist, 3),
                            "pocket_anchor_ref": round(ref_dist, 3)}
    out = qpos_robot.copy()
    for t, wrist in solved.items():
        out[t, cols] = (1 - w[t]) * qpos_robot[t, cols] + w[t] * wrist
    return out, {"hand": hand, "mode": "ik_retargeted",
                 "yaw_off": round(float(best[0]), 3),
                 "ik_ori_err": round(ori, 3),
                 "pocket_anchor_err": round(dist, 3),
                 "pocket_lateral_err": round(lat, 3)}


# -------------------------------------------------------------- scene --

def generate_pole_scene_xml(spec: PoleSpec, mass: float,
                            rest_pos: np.ndarray,
                            rest_quat: np.ndarray) -> str:
    """Template surgery: the largebox becomes the pole object. The handle
    capsule keeps the ``largebox_geom`` name, so all 22 hand-object pairs
    apply unchanged; the base primitive stands the object up and gets its
    own floor pair; the mesh is visual-only, placed by the sidecar."""
    xml = assets.robot_xml()
    sc = spec.mesh_scale
    mesh = (f'<mesh name="pole_object" file="{spec.obj_file}" '
            f'scale="{sc:.6f} {sc:.6f} {sc:.6f}" />')
    xml = mjcf.add_assets(xml, [mesh])

    r = spec.handle_radius
    z_lo, z_hi = spec.handle_z
    half_len = max((z_hi - z_lo) / 2, 0.02)
    b = spec.base
    if b["kind"] == "box":
        bh, bc = b["half"], b["center"]
        base_geom = (f'<geom name="largebox_base" type="box" '
                     f'size="{bh[0]:.4f} {bh[1]:.4f} {bh[2]:.4f}" '
                     f'pos="{bc[0]:.4f} {bc[1]:.4f} {bc[2]:.4f}" '
                     f'class="visual" material="black" rgba="0 0 0 0" />')
    else:
        base_geom = (f'<geom name="largebox_base" type="cylinder" '
                     f'size="{b["radius"]:.4f} {b["half_height"]:.4f}" '
                     f'pos="0 0 {b["half_height"]:.4f}" '
                     f'class="visual" material="black" rgba="0 0 0 0" />')
    inertia = mass * spec.inertia_per_mass
    p, q, mp, mq = rest_pos, rest_quat, spec.mesh_pos, spec.mesh_quat
    body = f'''<body name="largebox" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"
      quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}">
      <freejoint name="largebox_root" />
      <inertial mass="{mass:.3f}"
        pos="{spec.com[0]:.4f} {spec.com[1]:.4f} {spec.com[2]:.4f}"
        diaginertia="{inertia[0]:.3e} {inertia[1]:.3e} {inertia[2]:.3e}" />
      <geom name="largebox_geom" type="capsule"
        size="{r:.4f} {half_len:.4f}" pos="0 0 {(z_lo + z_hi) / 2:.4f}"
        class="visual" material="black" rgba="0 0 0 0" />
      {base_geom}
      <geom name="largebox_visual" type="mesh" mesh="pole_object"
        pos="{mp[0]:.4f} {mp[1]:.4f} {mp[2]:.4f}"
        quat="{mq[0]:.6f} {mq[1]:.6f} {mq[2]:.6f} {mq[3]:.6f}"
        class="visual" material="black" rgba="0.55 0.42 0.30 1" />
    </body>'''
    xml = mjcf.replace_object_body(xml, body)
    xml = mjcf.insert_pairs(xml, [
        '<pair name="largebox_base_floor" geom1="largebox_base" '
        'geom2="floor" solref="0.008 1" friction="1 1" condim="3" />'])
    xml = mjcf.harden_hand_object_pairs(xml)
    return mjcf.size_contact_buffers(xml)


# --------------------------------------------------------------- trial --

def _close_hand(model, qpos: np.ndarray, hand: str, grab: int, end: int,
                wrap: float, hold_c: float,
                caps: Optional[Dict[str, float]],
                released: bool) -> None:
    """Curl one hand around the handle in the reference (BrainCo only):
    pre-shape to the calibrated touch closure through the grab, then
    squeeze on the wrap preload; an early release opens back up."""
    T = len(qpos)
    t = np.arange(T)
    w1 = np.clip((t - (grab - CLOSE_LEAD)) / CLOSE_RAMP, 0.0, 1.0)
    w2 = np.clip((t - (grab + CLOSE_RAMP)) / CLOSE_RAMP, 0.0, 1.0)
    c = wrap * w1 + (hold_c - wrap) * w2
    if released:
        c = c * np.clip(1 - (t - end) / CLOSE_RAMP, 0.0, 1.0)
    for adr, angle in _closure_profile(model, hand, c,
                                       POWER_CLOSE_FRACTION,
                                       wrap=wrap, caps=caps).items():
        qpos[:, adr] = angle


FINGERS = ("index", "middle", "ring", "pinky", "thumb")
CONFORM_MARGIN = 0.001   # m: gap a backed-off finger keeps to the handle
CONFORM_SMOOTH = 9       # frames of blur over a conformed curl
CONFORM_TOUCH = 0.003    # m: a finger this close is holding the handle;
                         # the scan takes the first scale in the band
CONFORM_SETTLE = 9       # final blur over the repaired curl, so the hand
                         # does not tremble frame to frame
CONFORM_CEIL = 2.5       # most a conformed finger may tighten past the
                         # calibrated curl before the joint stops take over
CONFORM_OPEN = 0.05      # most open pose the scan considers when the
                         # calibrated curl is itself slack
CONFORM_HOLD = 0.30      # rad of proximal flexion a conformed finger always
                         # keeps: past it the hand stops reading as a grip,
                         # and a handle only clearable by opening that far
                         # is inside the palm — the placement's doing, not
                         # the curl's. Overlap there, hold always.


def _segment_gap(model, data, gid: int, hp: np.ndarray, hq: np.ndarray,
                 hr: float) -> float:
    """Surface gap between capsule geom `gid` and the handle segment
    (hp, hq, radius hr); negative means the two overlap."""
    axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
    half = float(model.geom_size[gid, 1])
    p, q = data.geom_xpos[gid] - half * axis, data.geom_xpos[gid] + half * axis
    u, v, w = q - p, hq - hp, p - hp
    a, b, c = u @ u, u @ v, v @ v
    d, e = u @ w, v @ w
    den = a * c - b * b
    if den < 1e-12:                      # parallel: clamp on one segment
        s, t = 0.0, (e / c if c > 1e-12 else 0.0)
    else:
        s, t = (b * e - c * d) / den, (a * e - b * d) / den
    s, t = min(max(s, 0.0), 1.0), min(max(t, 0.0), 1.0)
    close = np.linalg.norm((p + s * u) - (hp + t * v))
    return float(close - float(model.geom_size[gid, 0]) - hr)


def _conform_fingers(model, qpos: np.ndarray, hand: str,
                     grab: int, end: int,
                     gap_fn: Optional[Callable] = None,
                     hold: float = CONFORM_HOLD,
                     full_scan: bool = False) -> Dict[str, float]:
    """Close each finger onto the handle where it actually is. Returns
    the per-finger worst gap before the pass, in mm.

    `gap_fn(model, data, finger_gids) -> surface gap` swaps the object:
    the default measures against the largebox_geom handle capsule; the
    chair passes its hulls (chair._hull_gap_fn). `hold` is the proximal
    flexion a conformed finger always keeps (CONFORM_HOLD; 0 lets it
    open fully). `full_scan` drops the handle's assumption that opening
    clears and closing cuts in: a hand whose palm sits off the wood
    pokes the far member with OPEN fingers and clears it by curling, so
    every scale is probed and the most open touching pose wins, else
    the clear pose nearest to touching, else the least overlap — never
    the calibrated curl buried in the wood, which cages the object.

    calibrate_grasp caps each finger where it meets a cylinder at the
    CALIBRATION pocket, but the wrist re-aim, the pole's own drift and
    _clear_palms all move the handle afterwards — on an OMOMO
    clothes-stand carry the same hand ran 18 mm INSIDE the handle at the
    grab and 15 mm clear of it sixty frames later.

    So each finger is scaled — capped joints plus the distal joints that
    follow them — until its capsules just meet the handle: opened where
    the curl closes through it, closed further (past the calibration, up
    to the joint limits) where the handle moved out of reach. Fingers
    only: the palm rides on the arm, so moving it means re-solving the
    wrist IK."""
    data = mujoco.MjData(model)
    if gap_fn is None:
        handle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                   "largebox_geom")
        if handle < 0:
            return {}
        hr = float(model.geom_size[handle, 0])
        hh = float(model.geom_size[handle, 1])

        def gaps(finger_gids) -> float:
            axis = data.geom_xmat[handle].reshape(3, 3)[:, 2]
            hp = data.geom_xpos[handle] - hh * axis
            hq = data.geom_xpos[handle] + hh * axis
            return min(_segment_gap(model, data, g, hp, hq, hr)
                       for g in finger_gids)
    else:
        def gaps(finger_gids) -> float:
            return gap_fn(model, data, finger_gids)

    follow: Dict[int, list] = {}
    for eq in range(model.neq):
        if model.eq_type[eq] == mujoco.mjtEq.mjEQ_JOINT:
            follow.setdefault(int(model.eq_obj2id[eq]), []).append(
                (int(model.eq_obj1id[eq]), float(model.eq_data[eq, 1])))

    prefix = f"{hand[0]}h_"
    groups = {}
    for finger in FINGERS:
        gids = [g for g in range(model.ngeom)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                    or "").startswith(prefix + finger)]
        adrs = []
        for stem, _ in POWER_CLOSE_FRACTION.items():
            if not stem.startswith(finger):
                continue
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{hand}_{stem}_joint")
            if jid < 0:
                continue
            adrs.append((int(model.jnt_qposadr[jid]),
                         float(model.jnt_range[jid, 0]),
                         float(model.jnt_range[jid, 1])))
            adrs += [(int(model.jnt_qposadr[dj]),
                      float(model.jnt_range[dj, 0]),
                      float(model.jnt_range[dj, 1]))
                     for dj, _ in follow.get(jid, [])]
        if gids and adrs:
            groups[finger] = (gids, adrs, adrs[0][0])   # proximal first
    if not groups:
        return {}

    def pose(adrs, row, scale) -> None:
        for adr, jlo, jhi in adrs:
            data.qpos[adr] = min(max(scale * row[adr], jlo), jhi)

    worst: Dict[str, float] = dict.fromkeys(groups, np.inf)
    lo, hi = max(0, grab), min(len(qpos), end + 1)
    base = qpos[lo:hi].copy()
    scales = {f: np.ones(hi - lo) for f in groups}
    chosen = {f: np.ones(hi - lo) for f in groups}   # the per-frame scan
    for i, t in enumerate(range(lo, hi)):
        for finger, (gids, adrs, prox) in groups.items():
            data.qpos[:] = qpos[t]
            mujoco.mj_kinematics(model, data)
            worst[finger] = min(worst[finger], gaps(gids))

            # Close from open until the finger first meets the handle.
            # The gap is not monotone in the curl — a fingertip arcs back
            # toward the palm past the tangent, so two poses touch: resting
            # ON the surface, and a fist curled past it. Scanning upward
            # takes the first (the grip); a bisection lands on either.
            def probe(scale: float) -> float:
                pose(adrs, base[i], scale)
                mujoco.mj_kinematics(model, data)
                return gaps(gids)

            # start the scan at whatever scale still leaves CONFORM_HOLD
            # of proximal flexion
            held = abs(base[i, prox])
            opened = (min(1.0, hold / held) if held > 1e-6
                      else CONFORM_OPEN)
            grid = np.linspace(min(opened, 1.0), CONFORM_CEIL, 25)
            if full_scan:
                gs = np.array([probe(c) for c in grid])
                touching = np.nonzero((gs >= CONFORM_MARGIN)
                                      & (gs <= CONFORM_TOUCH))[0]
                clear = np.nonzero(gs > CONFORM_TOUCH)[0]
                if len(touching):
                    scales[finger][i] = grid[touching[0]]
                elif len(clear):
                    scales[finger][i] = grid[clear[np.argmin(gs[clear])]]
                else:
                    scales[finger][i] = grid[int(np.argmax(gs))]
                chosen[finger][i] = scales[finger][i]
                continue
            touch, reach, reach_gap = None, None, np.inf
            for cand in grid:
                g = probe(cand)
                if g < CONFORM_MARGIN:
                    break            # from here on the curl cuts in
                if g <= CONFORM_TOUCH:
                    touch = cand
                    break
                if g < reach_gap:    # nothing reaches yet: keep the closest
                    reach, reach_gap = cand, g
            if touch is None and reach is not None:
                step = grid[1] - grid[0]
                for cand in np.linspace(reach, min(reach + step,
                                                   CONFORM_CEIL), 9):
                    if probe(cand) < CONFORM_MARGIN:
                        break
                    touch = cand
            if touch is None:
                continue             # nowhere on the curl clears the handle
            scales[finger][i] = touch

    # each frame was solved on its own, so the raw scales jitter
    for finger, (gids, adrs, prox) in groups.items():
        s = scales[finger]
        if np.allclose(s, 1.0):
            continue
        if len(s) >= CONFORM_SMOOTH:
            s = smooth(s, CONFORM_SMOOTH)
            # the blur can hand a frame more curl than it had room for:
            # walk any frame that now cuts in back to the surface
            for i in range(len(s)):
                data.qpos[:] = qpos[lo + i]
                pose(adrs, base[i], s[i])
                mujoco.mj_kinematics(model, data)
                if gaps(gids) >= CONFORM_MARGIN:
                    continue
                held = abs(base[i, prox])
                low = (min(1.0, hold / held) if held > 1e-6
                       else CONFORM_OPEN)
                high = max(s[i], low)
                for _ in range(10):
                    mid = 0.5 * (low + high)
                    pose(adrs, base[i], mid)
                    mujoco.mj_kinematics(model, data)
                    if gaps(gids) >= CONFORM_MARGIN:
                        low = mid
                    else:
                        high = mid
                s[i] = low
            s = smooth(s, CONFORM_SETTLE)   # the repair spikes; ease them
            if full_scan:
                # the smoothing may have curled a finger back into the
                # wood where the scan had it clear: take the scan's own
                # choice there, a small step over a buried finger
                for i in range(len(s)):
                    data.qpos[:] = qpos[lo + i]
                    pose(adrs, base[i], s[i])
                    mujoco.mj_kinematics(model, data)
                    if gaps(gids) < CONFORM_MARGIN - CONFORM_TOUCH:
                        s[i] = chosen[finger][i]
        for adr, jlo, jhi in adrs:
            qpos[lo:hi, adr] = np.clip(s * base[:, adr], jlo, jhi)
    return {f: round(float(g) * 1000.0, 1) for f, g in worst.items()}


PALM_CLEAR_MARGIN = 0.004   # m: gap the handle keeps to a holding palm
PALM_CLEAR_SMOOTH = 9       # frames of smoothing on the clearing offset


def _box_escape(p: np.ndarray, box_pos: np.ndarray, box_mat: np.ndarray,
                box_half: np.ndarray, reach: float):
    """(depth, unit direction) that lifts a point of a capsule of radius
    `reach` out of the given box, or None when it is already clear."""
    local = box_mat.T @ (p - box_pos)
    clamped = np.clip(local, -box_half, box_half)
    v = p - (box_pos + box_mat @ clamped)
    d = float(np.linalg.norm(v))
    if d > 1e-9:
        depth = reach - d
        return (depth, v / d) if depth > 0 else None
    slack = box_half - np.abs(local)          # inside: leave by the near face
    k = int(np.argmin(slack))
    n = box_mat[:, k] * (1.0 if local[k] >= 0 else -1.0)
    return float(slack[k]) + reach, n


def _clear_palms(model, qpos: np.ndarray, hands, windows,
                 margin: float = PALM_CLEAR_MARGIN):
    """Slide the handle out of any palm it passes through. Returns the
    per-frame offset (T, 3) and the deepest overlap it had to clear.

    The pole is fitted to both hands' pocket tracks, and where those
    disagree about a rigid rod the fit splits the difference and the
    handle ends up inside a palm. Across 25 OMOMO clothes-stand clips
    every one buried a palm or finger 20-27 mm at its worst frame, on a
    pocket that leaves 4 mm of room for an 18.6 mm handle.

    The pole's version of box_carry's `optimize_box_placement`, per
    frame rather than once since a carried pole moves with the hands.
    The offset is perpendicular to the axis (sliding along it changes
    the grip without separating anything), smoothed for continuity, and
    floored so the base keeps its ground clearance."""
    handle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                               "largebox_geom")
    if handle < 0:
        return None, 0.0
    r = float(model.geom_size[handle, 0])
    half = float(model.geom_size[handle, 1])
    palms = []
    for hand in hands:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                f"{hand[0]}h")
        if gid >= 0:
            palms.append((gid, windows[hand]))
    if not palms:
        return None, 0.0

    data = mujoco.MjData(model)
    T = len(qpos)
    delta = np.zeros((T, 3))
    worst = 0.0
    for t in range(T):
        active = [g for g, (s, e) in palms if s <= t <= e]
        if not active:
            continue
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        axis = data.geom_xmat[handle].reshape(3, 3)[:, 2]
        centre = data.geom_xpos[handle]
        push = np.zeros(3)
        for _ in range(3):                     # a few passes: two palms
            for gid in active:                 # can ask for different pushes
                box_pos = data.geom_xpos[gid]
                box_mat = data.geom_xmat[gid].reshape(3, 3)
                box_half = model.geom_size[gid]
                deepest = None
                for s in np.linspace(-half, half, 13):
                    hit = _box_escape(centre + s * axis + push, box_pos,
                                      box_mat, box_half, r + margin)
                    if hit and (deepest is None or hit[0] > deepest[0]):
                        deepest = hit
                if deepest is None:
                    continue
                depth, n = deepest
                worst = max(worst, depth - margin)
                push = push + depth * n
        delta[t] = push - (push @ axis) * axis   # keep it off the axis

    if not np.any(delta):
        return None, 0.0
    delta = smooth3(delta, PALM_CLEAR_SMOOTH)
    # never push the base below the clearance a carried pole keeps
    low = qpos[:, -5] + delta[:, 2] - BASE_CLEAR
    delta[:, 2] -= np.minimum(low, 0.0)
    return delta, worst


class PoleTask:
    """pole as a ReconTask (recon.run)."""

    name = "pole"

    def detect(self, qpos, meta, params, options) -> Interaction:
        info = detect_pole_hold(meta, qpos)
        info = apply_hand_windows(info, meta, qpos,
                                  left=str(params["left"]),
                                  right=str(params["right"]))
        obj = infer_object(Path(meta["file_path"]).stem, str(params["object"]))
        # a hand joining late or releasing early shows its own window
        hands = "+".join(
            h + ("" if g == info.grab_frame and e == info.end_frame
                 else f"@f{g}-f{e}")
            for h, g, e in zip(info.hands, info.hand_grabs, info.ends))
        detail = (f"{hands} hold f{info.grab_frame}-f{info.end_frame} {obj} "
                  f"transport {info.transport:.2f} "
                  f"flags [{','.join(info.quality_flags) or '-'}]")
        return Interaction(info.ok, detail, "no pole hold", info=info,
                           object=obj)

    def build(self, qpos_robot, meta, inter: Interaction, params) -> Built:
        """The scene and reference of one pole trial. mass=0 uses the
        spec's default; grasp_close overrides the calibrated touch
        closure; grasp is the wrist treatment ("auto" = ik_retargeted
        where the hand model exists, falling back per hand; "reference"
        keeps the clip's own wrist)."""
        info: PoleInfo = inter.info
        mass = float(params["mass"])
        grasp_close = float(params["grasp_close"])
        grasp = str(params["grasp"])
        if grasp not in GRASPS:
            raise SystemExit(f"unknown grasp {grasp!r} "
                             f"(use {', '.join(GRASPS)})")
        spec = load_spec(inter.object)
        mass = mass if mass > 0 else spec.mass
        diameter = 2 * spec.handle_radius
        fps = meta["fps"]
        T = len(qpos_robot)
        s, e = info.grab_frame, info.end_frame
        released = "released_early" in info.quality_flags

        calib = {h: calibrate_grasp(h, diameter, POWER_CLOSE_FRACTION,
                                    wrap_radius=spec.handle_radius)
                 for h in info.hands}
        grasp_mode = grasp if grasp != "auto" else "ik_retargeted"
        if any(c is None for c in calib.values()):
            grasp_mode = "reference"    # handless template: no wrist to re-aim

        # 1. object state from the ORIGINAL motion: the clip's own wrist
        # pockets place the pole, never the re-aimed ones
        pocket_tracks = [
            palm_grip_track(qpos_robot, h, spec.handle_radius, s,
                            calib[h][1] if calib[h] else None)
            for h in info.hands]

        jp = meta["joint_positions"]
        reach = pocket_tracks[0][s, :2] - jp[s, 0, :2]
        approach_yaw = (float(np.arctan2(reach[1], reach[0]))
                        if np.linalg.norm(reach) > 1e-6 else 0.0)

        obj_qpos, anchors, rest_pos, obj_flags = build_pole_trajectory(
            pocket_tracks, info, spec, fps, approach_yaw)

        # 2. wrist retarget onto the estimated object's own axis
        grasp_infos: List[Dict] = []
        grasp_flags: List[str] = []
        for hand, hand_grab, hand_end in zip(info.hands, info.hand_grabs,
                                             info.ends):
            if grasp_mode == "ik_retargeted":
                qpos_robot, ginfo = retarget_pole_hand(
                    qpos_robot, meta, info, hand, calib[hand][1], obj_qpos,
                    anchors[hand], grab=hand_grab, end=hand_end)
                if ginfo["mode"] == "reference":
                    grasp_flags.append(f"ik_fallback_{hand}")
            else:
                ginfo = {"hand": hand, "mode": "reference"}
                if calib[hand] is None:
                    ginfo["fallback"] = "no_hand_model"
            grasp_infos.append(ginfo)

        scene_xml = generate_pole_scene_xml(spec, mass, obj_qpos[0, :3],
                                            obj_qpos[0, 3:])
        model = mujoco.MjModel.from_xml_string(scene_xml)
        layout.check_scene(model)
        qpos = assemble_qpos(model, qpos_robot, obj_qpos)

        closures: Dict[str, Dict] = {}
        for hand, hand_grab, hand_end in zip(info.hands, info.hand_grabs,
                                             info.ends):
            if calib[hand] is None:
                closures[hand] = {"touch": 0.0, "hold": 0.0, "caps": None}
                continue
            wrap = grasp_close if grasp_close > 0 else calib[hand][0]
            hold_c = wrap + WRAP_SQUEEZE
            caps = calib[hand][2]
            _close_hand(model, qpos, hand, hand_grab, hand_end, wrap, hold_c,
                        caps, hand_end < T - 6)

        # Wrists and curl are final, so the handle can now be settled into
        # the hands. The passes alternate because they push opposite ways:
        # sliding the handle off a palm drives it into the fingertips curled
        # around the far side. Two rounds land it in the free space between.
        windows = {h: (g, e2) for h, g, e2 in zip(info.hands, info.hand_grabs,
                                                  info.ends)}
        palm_clear, shift, worst_pen = None, np.zeros(3), 0.0
        pre_gaps: Dict[str, Dict] = {}
        for round_ in range(2):
            delta, palm_pen = _clear_palms(model, qpos, info.hands, windows)
            worst_pen = max(worst_pen, palm_pen)
            if delta is not None:
                obj_qpos = obj_qpos.copy()
                obj_qpos[:, :3] += delta
                qpos[:, -7:-4] = obj_qpos[:, :3]
                scene_xml = generate_pole_scene_xml(spec, mass, obj_qpos[0, :3],
                                                    obj_qpos[0, 3:])
                model = mujoco.MjModel.from_xml_string(scene_xml)
                shift = np.maximum(shift, np.abs(delta).max(axis=0))
            for hand, hand_grab, hand_end in zip(info.hands, info.hand_grabs,
                                                 info.ends):
                if calib[hand] is None:
                    continue
                gaps = _conform_fingers(model, qpos, hand, hand_grab, hand_end)
                if round_ == 0:          # the record is of the curl as calibrated
                    pre_gaps[hand] = gaps
        if worst_pen > 0:
            palm_clear = {"max_pen_before_mm": round(float(worst_pen) * 1000, 1),
                          "max_shift_mm": round(float(np.linalg.norm(shift))
                                                * 1000, 1)}

        for hand, hand_grab, hand_end in zip(info.hands, info.hand_grabs,
                                             info.ends):
            if calib[hand] is None:
                continue
            wrap = grasp_close if grasp_close > 0 else calib[hand][0]
            hold_c = wrap + WRAP_SQUEEZE
            caps = calib[hand][2]
            closures[hand] = {
                "touch": round(wrap, 4), "hold": round(hold_c, 4),
                "caps": ({k: round(float(v), 3) for k, v in caps.items()}
                         if caps else None),
                # worst finger-vs-handle gap the calibrated curl reached
                # before _conform_fingers backed it off (mm, negative =
                # through the handle)
                "pre_conform_gap_mm": pre_gaps.get(hand) or None}

        p1 = pocket_tracks[0]
        lift_height = float(p1[s:e + 1, 2].max() - p1[s, 2])

        # ends held: pad with the frozen final pose so the receding horizon
        # has a full reference window (see pick.py)
        pad = 0 if released else int(HOLD_PAD_S * fps)

        tspec = TrialSpec(
            task_type="pole_carry",
            object=ObjectSpec(
                kind="mesh", size=[diameter, diameter, spec.height],
                mass=mass, symmetry="axial",
                info={"name": spec.name, "handle_radius": spec.handle_radius,
                      "height": spec.height}),
            window=(s, e),
            starts_held=info.starts_held,
            lift_height=lift_height,
            # per-hand grip anchors for the solve: hold the SIMULATED pole
            # so the point `anchor` up its axis sits in each palm's pocket
            grips=[Grip(
                hand=h, window=(g, e2),
                pocket=calib[h][1] if calib[h] else None,
                anchor=[0.0, 0.0, anchors[h]],
                extra={"closure_touch": closures[h]["touch"],
                       "closure_hold": closures[h]["hold"],
                       "closure_caps": closures[h]["caps"],
                       "pre_conform_gap_mm":
                           closures[h].get("pre_conform_gap_mm")})
                for h, g, e2 in zip(info.hands, info.hand_grabs, info.ends)],
            grasp=grasp_infos,
            flags=info.quality_flags + obj_flags + grasp_flags,
            hand_noise_scale=HAND_NOISE_SCALE,
            extra={"hands": list(info.hands),
                   "palm_clearance": palm_clear,
                   "transport": round(info.transport, 3)},
        )
        # contact reference and interaction graph: the driver's defaults
        # (each holding palm at its own FK track over its window)
        return Built(scene_xml, model, qpos, tspec, pad=pad)

    def describe(self, task_info: Dict) -> str:
        clear = task_info.get("palm_clearance") or {}
        if not clear:
            return ""
        return (f" palm-clear (pen {clear['max_pen_before_mm']:.0f}mm "
                f"cleared, shift {clear['max_shift_mm']:.0f}mm)")


TASK = PoleTask()


def emit_pole_trial(
    qpos_robot: np.ndarray,
    meta: Dict,
    info: PoleInfo,
    out_root: Path,
    task: str,
    data_id: int = 0,
    object_name: str = "floorlamp",
    mass: float = 0.0,
    grasp_close: float = 0.0,
    grasp: str = "auto",
) -> Optional[Path]:
    """Write a complete pole trial from an already detected hold (the
    recon.run pipeline from `build` on). Returns the trial dir, or None
    when the motion admits no scene."""
    from .run import emit  # late: run imports this module's siblings
    params = {"object": object_name, "mass": mass, "grasp_close": grasp_close,
              "grasp": grasp, "left": "auto", "right": "auto"}
    trial, _info, _why = emit(TASK, qpos_robot, meta,
                              Interaction(True, "", info=info,
                                          object=object_name),
                              out_root, task, params, data_id=data_id)
    return trial
