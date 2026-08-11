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
import hashlib
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


def read_prompt_files(paths) -> list:
    """One prompt per line; '#' comments and blank lines dropped. Keeps a
    banked prompt set under version control instead of in shell history."""
    out = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def _words(text: str) -> list:
    return "".join(c if c.isalnum() or c.isspace() else " "
                   for c in text.lower()).split()


def _slug(text: str) -> str:
    return "_".join(_words(text)[:6]) or "prompt"


def unique_slugs(texts: list) -> list:
    """One filename per prompt, readable and collision-free. Six words is
    usually enough, but prompts that share a prefix ("A person picks up a
    box from the floor ..." vs "... from low on their left side") collide,
    so the slug grows until they diverge. Two prompts identical for 16
    words fall back to a content hash — a silent overwrite would lose an
    embedding that costs a GPU job to recompute."""
    for n in range(6, 17):
        slugs = ["_".join(_words(t)[:n]) or "prompt" for t in texts]
        if len(set(slugs)) == len(slugs):
            return slugs
    return [f"{s}_{hashlib.sha256(t.encode()).hexdigest()[:6]}"
            if slugs.count(s) > 1 else s
            for s, t in zip(slugs, texts)]


def save_npz(path, texts, embeddings, lengths) -> None:
    np.savez(path, texts=np.array(list(texts)), embeddings=embeddings,
             lengths=np.array(list(lengths), dtype=np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_dir", default=None,
                    help="dir with motion_*/meta.json to sweep")
    ap.add_argument("--text", action="append", default=[],
                    help="prompt string to encode (repeatable)")
    ap.add_argument("--file", action="append", default=[],
                    help="file of prompts, one per line, '#' comments and "
                         "blank lines ignored (repeatable)")
    ap.add_argument("--url",
                    default=os.environ.get("TEXT_ENCODER_URL",
                                           DEFAULT_TEXT_ENCODER_URL))
    ap.add_argument("--output", default=None)
    ap.add_argument("--split", action="store_true",
                    help="write one snapshot per prompt, named by its text, "
                         "instead of a single combined file")
    args = ap.parse_args()

    prompts = collect_unique_prompts(args.batch_dir) if args.batch_dir else []
    prompts += [t for t in sanitize_texts(args.text + read_prompt_files(args.file))
                if t.strip()]
    prompts = list(dict.fromkeys(prompts))
    if not prompts:
        sys.exit("No prompts: pass --batch_dir, --text and/or --file")
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

    # batch_generate.py auto-detects ONE combined text_embeddings.npz beside
    # the motions, so a batch run always gets that file whatever --split says;
    # --split only reshapes our own snapshot store.
    if args.batch_dir:
        output_path = Path(args.output or os.path.join(args.batch_dir,
                                                       "text_embeddings.npz"))
        save_npz(output_path, prompts, embeddings, lengths)
        print(f"Saved {embeddings.shape} embeddings to {output_path}")

    if args.split:
        for i, name in enumerate(unique_slugs(prompts)):
            save_npz(snap_dir / f"{name}.npz", prompts[i:i + 1],
                     embeddings[i:i + 1], lengths[i:i + 1])
            print(f"  {name}.npz")
        print(f"{len(prompts)} snapshot(s) in {snap_dir}")
        return

    if args.batch_dir:
        snap = snap_dir / f"{Path(args.batch_dir).resolve().name}.npz"
        shutil.copy2(output_path, snap)
    else:
        snap = Path(args.output or snap_dir / f"{_slug(prompts[0])}.npz")
        save_npz(snap, prompts, embeddings, lengths)
        print(f"Saved {embeddings.shape} embeddings to {snap}")
    print(f"Snapshot for offline demo use: {snap}")


if __name__ == "__main__":
    main()
