"""Snapshot text embeddings for a batch dir via the tunneled encoder server.

Same output as kimodo's scripts/encode_prompts.py (batch_dir/
text_embeddings.npz, auto-detected by batch_generate.py), but fetched
through the remote encoder API instead of loading the 8B model locally —
one cheap call while the tunnel is already up, and the prompt never needs
the cluster again.

Usage (tunnel up, e.g. after `studio tunnel <node>`):
    .venv-kimodo/bin/python scripts/encode_via_api.py --batch_dir <dir>
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from kimodo.meta import load_prompts_from_meta
from kimodo.model.registry import DEFAULT_TEXT_ENCODER_URL
from kimodo.model.text_encoder_api import TextEncoderAPI
from kimodo.sanitize import sanitize_texts


def collect_unique_prompts(batch_dir: str) -> list:
    # sanitized exactly like the model does at generation time, matching
    # kimodo's encode_prompts.py
    prompts = []
    for d in sorted(os.listdir(batch_dir)):
        meta_path = os.path.join(batch_dir, d, "meta.json")
        if not (d.startswith("motion_") and os.path.isfile(meta_path)):
            continue
        texts, _ = load_prompts_from_meta(meta_path)
        prompts.extend(t for t in sanitize_texts(texts) if t.strip())
    return list(dict.fromkeys(prompts))


def _slug(text: str) -> str:
    words = "".join(c if c.isalnum() or c.isspace() else " "
                    for c in text.lower()).split()
    return "_".join(words[:6]) or "prompt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_dir", default=None,
                    help="dir with motion_*/meta.json to sweep")
    ap.add_argument("--text", action="append", default=[],
                    help="prompt string to encode (repeatable)")
    ap.add_argument("--url",
                    default=os.environ.get("TEXT_ENCODER_URL",
                                           DEFAULT_TEXT_ENCODER_URL))
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    prompts = collect_unique_prompts(args.batch_dir) if args.batch_dir else []
    prompts += [t for t in sanitize_texts(args.text) if t.strip()]
    prompts = list(dict.fromkeys(prompts))
    if not prompts:
        sys.exit("No prompts: pass --batch_dir and/or --text")
    for p in prompts:
        print(f"  - {p}")

    encoded, lengths = TextEncoderAPI(url=args.url)(prompts)
    embeddings = encoded.float().cpu().numpy()  # (N, L, D)

    # a zero embedding means the offline stand-in answered, not the real
    # encoder — banking it would poison the snapshot store
    zero = [prompts[i] for i in range(len(prompts))
            if not np.abs(embeddings[i, :lengths[i]]).sum() > 0]
    if zero:
        sys.exit(f"Server returned ZERO embeddings for {zero} — you are "
                 "talking to the offline stand-in server, not the real "
                 "encoder. Bring the tunnel up and retry.")

    snap_dir = Path(__file__).resolve().parents[1] / "offline_cache/snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_dir:
        output_path = args.output or os.path.join(args.batch_dir,
                                                  "text_embeddings.npz")
        snap = snap_dir / f"{Path(args.batch_dir).resolve().name}.npz"
    else:
        output_path = args.output or str(snap_dir / f"{_slug(prompts[0])}.npz")
        snap = Path(output_path)

    np.savez(
        output_path,
        texts=np.array(prompts),
        embeddings=embeddings,
        lengths=np.array(lengths, dtype=np.int64),
    )
    print(f"Saved {embeddings.shape} embeddings to {output_path}")
    if snap != Path(output_path):
        shutil.copy2(output_path, snap)
    print(f"Snapshot for offline demo use: {snap}")


if __name__ == "__main__":
    main()
