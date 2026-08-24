"""One prompt, several 2D root paths — scripted Kimodo generation.

The counterpart to drawing a root path by hand in the viser demo: the same
`Root2DConstraintSet` the demo builds from its "2D Root" track (see
kimodo.demo.generation.compute_model_constraints_lst), authored here as
sparse (frame, x, z) waypoints so a set of destinations is reproducible
from one command.

The prompt supplies the manipulation, the root path supplies the
navigation: every variant below picks the same box off the floor and
carries it somewhere else.

Runs in the kimodo venv, against the same text encoder the demo uses —
so with prompts already banked in offline_cache/, no cluster is needed:

    .venv-kimodo/bin/python scripts/offline_encoder.py \\
        --cache-dir offline_cache/embeddings --model kimodo-g1-rp \\
        --port 9550 $(printf -- '--seed %s ' offline_cache/snapshots/*.npz) &
    TEXT_ENCODER_MODE=api TEXT_ENCODER_URL=http://127.0.0.1:9550/ \\
    kimodo_EMBED_CACHE_DIR=$PWD/offline_cache/embeddings \\
        .venv-kimodo/bin/python scripts/gen_root_paths.py

`--samples K` generates K candidates per variant and keeps the one that
picks deepest while still tracking its path; each kept clip lands in
raw_motion/<prefix><variant>.npz with its path beside it as
<prefix><variant>.path.json (read back by scripts/render_root_paths.py).

Kimodo world coordinates: x lateral, z forward, y up.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kimodo.constraints import Root2DConstraintSet
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.model import load_model
from kimodo.tools import seed_everything

ROOT = Path(__file__).resolve().parents[1]
RAW_MOTION = ROOT / "raw_motion"

# "lifts a box off the ground and carries it forward" never produces a lift
# under this model — 24/24 samples stayed upright, and recon rejects them
# as no_lift. This phrasing picks reliably; the carry comes from the path.
PROMPT = ("A person bends down, picks up a box from the floor, "
          "and stands up straight holding it.")
T = 240  # 8 s @ 30 fps; frame indices must stay < T

# variant -> [(frame, x, z), ...]. Every path holds the root at the origin
# through the pick (~f50-80) and only then heads for its destination.
VARIANTS = {
    "fwd":   [(0, 0.0, 0.0), (100, 0.0, 0.10), (239, 0.0, 2.3)],
    "left":  [(0, 0.0, 0.0), (100, 0.0, 0.10), (180, -0.8, 1.0),
              (239, -2.0, 1.4)],
    "right": [(0, 0.0, 0.0), (100, 0.0, 0.10), (180, 0.8, 1.0),
              (239, 2.0, 1.4)],
    # an L: straight out, then a hard turn to the right
    "hook":  [(0, 0.0, 0.0), (100, 0.0, 0.10), (165, 0.0, 1.25),
              (205, 0.55, 1.85), (239, 1.65, 1.95)],
}

LH, RH = 25, 33  # kimodo hand tips


def score(joints: np.ndarray, waypoints) -> tuple:
    """(pick depth, worst waypoint error) for one sample. Lower hand height
    means a deeper pick; the error is how far the root missed its
    waypoints."""
    mid_hand_y = ((joints[:, LH] + joints[:, RH]) / 2)[:, 1]
    root = joints[:, 0]
    err = max(float(np.hypot(root[f, 0] - x, root[f, 2] - z))
              for f, x, z in waypoints)
    return float(mid_hand_y.min()), err


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=RAW_MOTION)
    ap.add_argument("--prefix", default="rp_")
    ap.add_argument("--samples", type=int, default=3,
                    help="candidates per variant; the best one is kept")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=100,
                    help="diffusion steps")
    ap.add_argument("--max-path-err", type=float, default=0.25,
                    help="m; candidates that miss a waypoint by more than "
                         "this are rejected before the pick-depth ranking")
    ap.add_argument("--variant", action="append", default=[],
                    help="restrict to these variants (repeatable)")
    ap.add_argument("--model", default="Kimodo-G1-RP-v1")
    args = ap.parse_args()

    variants = {k: v for k, v in VARIANTS.items()
                if not args.variant or k in args.variant}
    if not variants:
        raise SystemExit(f"no such variant(s): {args.variant}")
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device=device, default_family="Kimodo")
    skeleton = model.skeleton

    # one batched call: every (variant, candidate) pair is a batch element
    # carrying its own root-path constraint
    names, constraints = [], []
    for name, waypoints in variants.items():
        frames = torch.tensor([w[0] for w in waypoints])
        xz = torch.tensor([[w[1], w[2]] for w in waypoints],
                          dtype=torch.float32, device=device)
        for k in range(args.samples):
            names.append((name, k))
            constraints.append([Root2DConstraintSet(skeleton, frames, xz)])

    print(f"generating {len(names)} clip(s): {len(variants)} path(s) x "
          f"{args.samples} sample(s), {T} frames, seed {args.seed}",
          flush=True)
    seed_everything(args.seed)
    out = model([PROMPT] * len(names), [T] * len(names),
                num_denoising_steps=args.steps, constraint_lst=constraints,
                return_numpy=True)

    best: dict = {}
    for b, (name, k) in enumerate(names):
        joints = out["posed_joints"][b]
        hand_y, err = score(joints, variants[name])
        ok = err <= args.max_path_err
        print(f"  {name}[{k}]: pick_hand_y={hand_y:.2f} path_err={err:.2f}"
              f"{'' if ok else '  REJECTED (path)'}", flush=True)
        if ok and (name not in best or hand_y < best[name][1]):
            best[name] = (b, hand_y, err)

    if len(best) < len(variants):
        missing = sorted(set(variants) - set(best))
        print(f"no candidate tracked the path for: {', '.join(missing)} — "
              "re-run with more --samples or a gentler path", flush=True)

    for name, (b, hand_y, err) in sorted(best.items()):
        stem = f"{args.prefix}{name}"
        npz = args.out / f"{stem}.npz"
        save_kimodo_npz(str(npz), {
            "posed_joints": out["posed_joints"][b],
            "global_rot_mats": out["global_rot_mats"][b],
            "local_rot_mats": out["local_rot_mats"][b],
            "root_positions": out["posed_joints"][b][:, skeleton.root_idx, :],
            "foot_contacts": out["foot_contacts"][b],
        })
        # the renderer draws this on the floor; keep it in kimodo coords and
        # let the renderer permute, exactly like the clip itself
        (args.out / f"{stem}.path.json").write_text(json.dumps({
            "variant": name,
            "prompt": PROMPT,
            "num_frames": T,
            "seed": args.seed,
            "waypoints": [list(w) for w in variants[name]],
            "root_xz": out["posed_joints"][b][:, 0][:, [0, 2]].tolist(),
        }, indent=2) + "\n")
        print(f"kept {name} (pick {hand_y:.2f} m, path err {err:.2f} m) "
              f"-> {npz}", flush=True)


if __name__ == "__main__":
    main()
