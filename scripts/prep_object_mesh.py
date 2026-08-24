"""Measure the OMOMO pole-object meshes and write their sidecar specs.

The raw meshes are neither metric nor aligned (the floor lamp spans 4.65
units along an arbitrary axis). Rather than duplicating ~10 MB of
canonicalized meshes, this computes the affine that WOULD canonicalize
each one and stores it in a JSON sidecar next to the OBJ; scene
generation applies it through MuJoCo itself, referencing the raw OBJ in
place.

Canonical frame: +z along the handle, base at z=0, handle centered on
the z axis, metres.

The sidecar also carries everything reconstruction needs so it never
re-parses the OBJ: handle radius and extent, a base primitive (a disc,
or "box"), CoM and mass-normalized inertia, and a default mass.

Deterministic: re-running overwrites each sidecar with identical bytes.

    uv run python scripts/prep_object_mesh.py
"""

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = REPO_ROOT / "assets/object_mesh"

# per-object ground truth the mesh cannot supply: real height, default
# mass, which end rests on the floor ("wide"/"thin" by radial extent —
# every current object stands on its wide end, checked for the clothes
# stand against the OMOMO poses), and the base primitive kind
OBJECTS = {
    "floorlamp": {
        "file": "floorlamp_cleaned_simplified.obj",
        "height": 1.70, "mass": 3.0, "down": "wide", "base": "cylinder"},
    "tripod": {
        "file": "tripod_cleaned_simplified.obj",
        "height": 1.40, "mass": 1.5, "down": "wide", "base": "cylinder"},
    "clothesstand": {
        "file": "clothesstand_cleaned_simplified.obj",
        "height": 1.75, "mass": 2.5, "down": "wide", "base": "cylinder"},
}

N_SLICES = 40          # radial-profile resolution along the axis
HANDLE_R_MAX = 0.045   # m: a slice this thin (r90) counts as handle
GRAB_MARGIN = 0.05     # m: graspable band inset from the handle ends
END_BAND = 0.10        # fraction of the span probed at each end for the
                       # wide-end-down orientation rule


def load_obj(path: Path) -> np.ndarray:
    verts = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
    return np.array(verts)


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (w, x, y, z); loader's twin, inlined so the
    script runs without the studio package."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix taking unit vector a onto unit vector b."""
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def slice_r90(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """90th-percentile radial distance from the z axis per z slice."""
    r = np.hypot(p[:, 0], p[:, 1])
    out = np.zeros(len(edges) - 1)
    for i in range(len(out)):
        m = (p[:, 2] >= edges[i]) & (p[:, 2] < edges[i + 1])
        out[i] = np.percentile(r[m], 90) if m.sum() > 3 else np.inf
    return out


def canonicalize(name: str, cfg: dict) -> dict:
    v = load_obj(MESH_DIR / cfg["file"])

    # 1. principal axis to z (eigh ascending: column 2 is the long axis)
    mean = v.mean(0)
    c = v - mean
    _, evecs = np.linalg.eigh(c.T @ c / len(c))
    R = evecs[:, ::-1].T            # rows: major, mid, minor
    if np.linalg.det(R) < 0:
        R[2] *= -1.0
    # major -> z, mid -> x, minor -> y; a cyclic permutation, so
    # handedness survives
    A = R[[1, 2, 0]]
    b = -A @ mean
    p = v @ A.T + b

    # 2. which end is down: the configured end (by radial extent) at -z
    span = p[:, 2].max() - p[:, 2].min()
    lo_band = p[:, 2] < p[:, 2].min() + END_BAND * span
    hi_band = p[:, 2] > p[:, 2].max() - END_BAND * span
    r = np.hypot(p[:, 0] - np.median(p[:, 0]), p[:, 1] - np.median(p[:, 1]))
    wide_is_lo = np.percentile(r[lo_band], 98) > np.percentile(r[hi_band], 98)
    flip = wide_is_lo != (cfg["down"] == "wide")
    if flip:
        F = np.diag([1.0, -1.0, -1.0])
        A, b = F @ A, F @ b
        p = v @ A.T + b

    # 3. metric scale (applied now so thresholds below are in meters)
    scale = cfg["height"] / span
    A, b = scale * A, scale * b
    p = v @ A.T + b

    # 4. handle band from the radial profile, then re-align its own
    # centroid line onto z (PCA is biased by the head/base)
    def handle_band(p):
        edges = np.linspace(p[:, 2].min(), p[:, 2].max(), N_SLICES + 1)
        thin = slice_r90(p, edges) <= HANDLE_R_MAX
        runs, s = [], None
        for i in range(N_SLICES):
            if thin[i] and s is None:
                s = i
            if (not thin[i] or i == N_SLICES - 1) and s is not None:
                runs.append((s, i if thin[i] else i - 1))
                s = None
        if not runs:
            raise SystemExit(f"{name}: no handle-thin band found")
        s, e = max(runs, key=lambda r: r[1] - r[0])
        return float(edges[s]), float(edges[e + 1])

    for _ in range(2):
        z_lo, z_hi = handle_band(p)
        edges = np.linspace(z_lo, z_hi, 13)
        cents = []
        for i in range(12):
            m = (p[:, 2] >= edges[i]) & (p[:, 2] < edges[i + 1])
            if m.sum() > 3:
                cents.append(np.median(p[m], axis=0))
        cents = np.array(cents)
        d = cents[-1] - cents[0]
        d /= np.linalg.norm(d)
        Rd = rot_between(d, np.array([0.0, 0.0, 1.0]))
        A, b = Rd @ A, Rd @ b
        p = v @ A.T + b
        xy = np.median(cents @ Rd.T, axis=0)[:2]
        shift = np.array([xy[0], xy[1], p[:, 2].min()])
        b = b - shift
        p = p - shift

    z_lo, z_hi = handle_band(p)
    z_hi = min(z_hi, float(p[:, 2].max()))
    m_handle = (p[:, 2] >= z_lo) & (p[:, 2] < z_hi)
    handle_r = float(np.median(np.hypot(p[m_handle, 0], p[m_handle, 1])))
    height = float(p[:, 2].max())

    # 5. base primitive from the vertices below the handle
    m_base = p[:, 2] < max(z_lo, 0.02)
    if not m_base.any():
        m_base = p[:, 2] < 0.05 * height
    bp = p[m_base]
    if cfg["base"] == "box":
        lo98, hi98 = (np.percentile(bp, 2, axis=0),
                      np.percentile(bp, 98, axis=0))
        base = {"kind": "box",
                "half": [round(float(x), 4)
                         for x in np.maximum((hi98 - lo98) / 2, 0.01)],
                "center": [round(float(x), 4) for x in (hi98 + lo98) / 2]}
    else:
        br = float(np.percentile(np.hypot(bp[:, 0], bp[:, 1]), 98))
        bh = float(np.clip(np.percentile(bp[:, 2], 95) / 2, 0.01, 0.05))
        base = {"kind": "cylinder", "radius": round(max(br, 0.03), 4),
                "half_height": round(bh, 4),
                "center": [0.0, 0.0, round(bh, 4)]}

    # 6. CoM and mass-normalized inertia from the vertex cloud, surface
    # points weighted equally
    com = p.mean(0)
    q = p - com
    inertia = [float(np.mean(q[:, 1] ** 2 + q[:, 2] ** 2)),
               float(np.mean(q[:, 0] ** 2 + q[:, 2] ** 2)),
               float(np.mean(q[:, 0] ** 2 + q[:, 1] ** 2))]

    # decompose the affine for MuJoCo: p = A v + b with A = scale * R
    Rm = A / scale
    assert np.allclose(Rm @ Rm.T, np.eye(3), atol=1e-8)
    quat = rotmat_to_quat_wxyz(Rm)

    return {
        "name": name,
        "obj_file": cfg["file"],
        "mesh_scale": round(scale, 6),
        "mesh_quat": [round(float(x), 6) for x in quat],
        "mesh_pos": [round(float(x), 4) for x in b],
        "height": round(height, 4),
        "handle_radius": round(handle_r, 4),
        "handle_z": [round(z_lo, 4), round(z_hi, 4)],
        "grab_z": [round(z_lo + GRAB_MARGIN, 4),
                   round(z_hi - GRAB_MARGIN, 4)],
        "base": base,
        "mass": cfg["mass"],
        "com": [round(float(x), 4) for x in com],
        "inertia_per_mass": [round(x, 5) for x in inertia],
    }


def main() -> None:
    for name, cfg in OBJECTS.items():
        if not (MESH_DIR / cfg["file"]).is_file():
            # the roster is whatever meshes are present, so a removed
            # mesh simply drops its object
            print(f"{name:13s} skipped ({cfg['file']} not present)")
            continue
        spec = canonicalize(name, cfg)
        out = MESH_DIR / f"{name}.json"
        out.write_text(json.dumps(spec, indent=2) + "\n")
        print(f"{name:13s} h={spec['height']:.2f} "
              f"handle r={spec['handle_radius']*100:.1f}cm "
              f"z=[{spec['handle_z'][0]:.2f},{spec['handle_z'][1]:.2f}] "
              f"base={spec['base']['kind']} -> {out.name}")


if __name__ == "__main__":
    main()
