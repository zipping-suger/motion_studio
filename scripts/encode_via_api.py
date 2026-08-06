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
import sys

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_dir", required=True)
    ap.add_argument("--url",
                    default=os.environ.get("TEXT_ENCODER_URL",
                                           DEFAULT_TEXT_ENCODER_URL))
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    output_path = args.output or os.path.join(args.batch_dir,
                                              "text_embeddings.npz")
    prompts = collect_unique_prompts(args.batch_dir)
    if not prompts:
        sys.exit(f"No prompts found in {args.batch_dir}")
    for p in prompts:
        print(f"  - {p}")

    encoded, lengths = TextEncoderAPI(url=args.url)(prompts)
    embeddings = encoded.float().cpu().numpy()  # (N, L, D)

    np.savez(
        output_path,
        texts=np.array(prompts),
        embeddings=embeddings,
        lengths=np.array(lengths, dtype=np.int64),
    )
    print(f"Saved {embeddings.shape} embeddings to {output_path}")


if __name__ == "__main__":
    main()
