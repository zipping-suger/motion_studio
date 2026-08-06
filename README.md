# kimodo_studio

Interactive loop: author a single G1 motion clip in Kimodo's viser demo,
then get a **dynamically feasible robot+object trajectory** out of
[mppi_locoma](../humanoid_project/experiments/mppi_locoma)'s scene
reconstruction + SPIDER retargeting.

```
 browser (viser demo :7860)          this repo                  mppi_locoma (.venv)
┌──────────────────────────┐   ┌──────────────────────┐   ┌─────────────────────────┐
│ prompt + constraints     │   │ studio watch         │   │ build_trial.py          │
│ → generate → pick sample │──▶│  shim + manifest     │──▶│ run_mjwp.py (MPPI, GPU) │
│ → Save Example           │   │  runs/<name>/        │   │ eval_trials.py          │
└──────────────────────────┘   └──────────────────────┘   └─────────────────────────┘
        ▲ text embeddings via SSH tunnel :9550 ── LLM2Vec-8B on the cluster
```

The text encoder (LLM2Vec-Llama-3-8B, ~16 GB VRAM) runs **remotely**;
the demo's diffusion model and the retargeting run **locally**.

## Setup (once)

```bash
uv sync                 # this repo's tiny venv (the `studio` CLI)
uv run studio setup     # dedicated kimodo venv: torch + kimodo + viser fork
```

Notes:
- First `studio demo` downloads `Kimodo-G1-RP-v1` from HuggingFace and builds
  the viser web client (downloads Node 20) — a few minutes, once.
- All paths/hosts live in `config.yml`.

## Daily loop

```bash
# on the cluster (once per session):
sbatch remote/encoder_euler.sh          # job log prints "encoder node: <node>"

# terminal 1 — tunnel (stays open):
uv run studio tunnel <node>

# terminal 2 — demo:
uv run studio demo                      # http://127.0.0.1:7860

# terminal 3 — pipeline:
uv run studio watch
```

In the browser: write a prompt (and optional timeline constraints), generate,
**click the sample you like** (with `num_samples > 1` the save buttons stay
disabled until you commit a sample by clicking it), then **Save Example**.
The watcher picks it up, reconstructs the scene, retargets, and prints the
LIFT / DynaRetarget verdict.

```bash
uv run studio list                      # runs + verdicts
uv run studio view <name>               # viser: reference vs. result (:8081)
uv run studio run <example-dir>         # process one example without watching
uv run studio promote <name>            # copy a good trial into mppi_locoma's
                                        # central outputs (paper dataset flow)
```

Each run is self-contained under `runs/<name>/`: shimmed inputs,
`outputs/` (scene.xml, reference + retargeted trajectories), `logs/`, and
`manifest.json` (prompt, seed, git SHAs of all three repos, mppi_locoma
config snapshot, verdict). Nothing reaches the paper dataset unless you
`studio promote` it — `export_dataset.py` sweeps every passing trial in the
central outputs, which is exactly why studio runs are kept isolated.

## Gotchas

- **Tunnel must be up before `studio demo`**: at startup the demo prewarms
  embeddings for all example prompts. `studio demo` pre-checks the tunnel and
  tells you what to start if it's down. Repeat prompts are free afterwards
  (`~/.cache/kimodo_demo/embeddings`).
- **VRAM**: demo + retargeting share the 16 GB GPU. If retargeting OOMs while
  the demo is open, lower `retarget.num_samples` in `mppi_locoma/config.yml`
  (e.g. 2048 → 1024).
- **Reach-only clips are SKIPPED**: scene reconstruction infers the object
  from pick/carry hand kinematics; a motion with no object interaction has no
  scene to build. That's a verdict, not a bug.
- **Retarget hyperparameters** are single-sourced in
  `mppi_locoma/config.yml` (`retarget:` section), same as its own
  `run_pipeline.sh`.
- The demo's Save Example dir is hardcoded inside the kimodo package
  (`kimodo/assets/demo/examples/kimodo-g1-rp/`); `studio watch` watches it in
  place, and saved examples reappear in the demo's Examples dropdown for
  re-loading/editing.
