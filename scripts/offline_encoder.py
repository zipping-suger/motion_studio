"""Offline text-encoder server for tunnel-free demo authoring.

Speaks the same gradio /DemoWrapper protocol as kimodo's
run_text_encoder_server.py, but instead of running the 8B model it looks
embeddings up in an ISOLATED offline cache, populated from
text_embeddings.npz snapshots (scripts/encode_via_api.py) and synced from
the demo's own cache. Unknown prompts get a ZERO embedding — generation
then ignores the text — plus a loud warning here; that is deliberate, so
the demo's startup prewarm cannot crash while offline. The isolated
cache dir also means zero entries can never leak into the real demo
cache used by tunnel sessions.

Runs in the kimodo venv; started automatically by `studio demo --offline`.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import gradio as gr
import numpy as np

from kimodo.demo.embedding_cache import EmbeddingCache
from kimodo.sanitize import sanitize_texts

# must match the demo's api-mode encoder class, which namespaces cache keys
ENCODER_ID = "TextEncoderAPI"
REAL_CACHE_ROOT = "~/.cache/kimodo_demo/embeddings"


def seed_from_npz(cache: EmbeddingCache, npz_paths) -> int:
    """Import encode_prompts.py-format snapshots. Uses EmbeddingCache's
    private key/save helpers on purpose — the cache layout is the contract
    with the demo, and the kimodo fork is pinned locally."""
    n = 0
    with cache._lock:
        cache._load_index()
        for p in npz_paths:
            data = np.load(p)
            for i, text in enumerate(data["texts"]):
                key = cache._make_key(sanitize_texts([str(text)])[0])
                if os.path.exists(cache._entry_path(key)):
                    continue
                length = int(data["lengths"][i])
                cache._disk_save(
                    key, data["embeddings"][i, :length].astype(np.float32))
                n += 1
        cache._save_index()
    return n


def sync_from_real_cache(cache: EmbeddingCache, real_root: str) -> int:
    src = Path(real_root).expanduser() / cache.model_name
    if not src.is_dir():
        return 0
    dst = Path(cache._model_dir())
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.glob("*.npy"):
        if not (dst / f.name).exists():
            shutil.copy2(f, dst / f.name)
            n += 1
    src_index = src / "index.json"
    if src_index.exists():
        with cache._lock:
            cache._load_index()
            cache._index = {**json.loads(src_index.read_text()), **cache._index}
            cache._save_index()
    return n


class DemoWrapper:
    # class name matters: gradio derives the /DemoWrapper endpoint name
    # from the callable's class, exactly like the real server
    def __init__(self, cache: EmbeddingCache, tmp_folder: str):
        self.cache = cache
        self.tmp_folder = tmp_folder
        self.dim = self._probe_dim()

    def _probe_dim(self) -> int:
        for f in Path(self.cache._model_dir()).glob("*.npy"):
            try:
                return int(np.load(f).shape[-1])
            except Exception:
                continue
        return 4096  # LLM2Vec llm_dim

    def __call__(self, text, filename, progress=gr.Progress()):
        sanitized = sanitize_texts([text])[0] if text and text.strip() else ""
        embedding = None
        if sanitized:
            embedding = self.cache._disk_load(self.cache._make_key(sanitized))
        if embedding is None:
            embedding = np.zeros((1, self.dim), dtype=np.float32)
            if sanitized:
                print(f"WARNING: no offline embedding for {text!r} — serving "
                      "ZEROS, generation will IGNORE this text. Encode it "
                      "once while a tunnel is up: scripts/encode_via_api.py",
                      flush=True)
        path = os.path.join(self.tmp_folder, filename)
        np.save(path, embedding)
        output_title = gr.Markdown(visible=True)
        output_text = gr.Markdown(visible=True, value=f"Text: {text}")
        download = gr.DownloadButton(visible=True, value=path)
        return download, output_title, output_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True,
                    help="isolated offline embedding cache root")
    ap.add_argument("--model", default="kimodo-g1-rp",
                    help="model short key (cache namespace)")
    ap.add_argument("--seed", action="append", default=[],
                    help="text_embeddings.npz snapshot(s) to import")
    ap.add_argument("--sync-from", default=REAL_CACHE_ROOT,
                    help="demo cache root to copy entries from")
    ap.add_argument("--port", type=int, default=9550)
    ap.add_argument("--tmp-folder", default="/tmp/offline_text_encoder/")
    args = ap.parse_args()

    cache = EmbeddingCache(model_name=args.model, encoder_id=ENCODER_ID,
                           base_dir=args.cache_dir)
    n_sync = sync_from_real_cache(cache, args.sync_from)
    n_seed = seed_from_npz(cache, args.seed)
    n_total = len(list(Path(cache._model_dir()).glob("*.npy"))) \
        if Path(cache._model_dir()).is_dir() else 0
    print(f"offline store: {n_total} embedding(s) "
          f"(+{n_sync} synced, +{n_seed} seeded)", flush=True)

    os.makedirs(args.tmp_folder, exist_ok=True)
    wrapper = DemoWrapper(cache, args.tmp_folder)

    # Mirror run_text_encoder_server.py's component/event layout so
    # gradio_client finds the identical /DemoWrapper endpoint.
    with gr.Blocks(title="Offline text encoder") as demo:
        gr.Markdown("# Offline text encoder (cached embeddings only)")
        text = gr.Textbox(label="Text prompt", type="text")
        btn = gr.Button("Encode", variant="primary")
        output_title = gr.Markdown("## Outputs", visible=False)
        output_text = gr.Markdown("", visible=False)
        download = gr.DownloadButton("Download", variant="primary",
                                     visible=False)
        filename = gr.Textbox(visible=False, value="embedding.npy")

        def clear_fn():
            return [gr.DownloadButton(visible=False),
                    gr.Markdown(visible=False), gr.Markdown(visible=False)]

        outputs = [download, output_title, output_text]
        gr.on(triggers=[text.submit, btn.click], fn=clear_fn,
              inputs=None, outputs=outputs).then(
            fn=wrapper, inputs=[text, filename], outputs=outputs)

    demo.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
