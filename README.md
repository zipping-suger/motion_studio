# kimodo_studio

Author a single G1 motion clip in Kimodo's viser demo, reconstruct the scene it
implies, and solve for a **dynamically feasible robot+object trajectory**.

```
 browser (viser demo :7860)         this repo                 solve venv (.venv-solve)
┌──────────────────────────┐   ┌──────────────────────┐   ┌─────────────────────────┐
│ prompt + constraints     │   │ studio.recon         │   │ studio.solve.loop       │
│ → generate → pick sample │──▶│  scene reconstruction│──▶│  SBMPC, 2048 worlds     │
│ → Save Example / Recon   │   │  shim + manifest     │   │  over SPIDER + warp     │
└──────────────────────────┘   └──────────────────────┘   └─────────────────────────┘
        ▲ embeddings via SSH tunnel :9550        studio.solve.evaluate → LIFT / DR
          LLM2Vec-8B on the cluster
```

studio owns both halves of the science: **scene reconstruction**
(`src/studio/recon/`, numpy + MuJoCo, no GPU) and the **reward stack and control
loop** of the solve (`src/studio/solve/`). SPIDER supplies the sampling
optimizer and the batched mujoco_warp simulator underneath, as a library.

## Requirements

| what | why | when |
|---|---|---|
| `kimodo` checkout | viser demo, diffusion model, motion NPZ I/O | run time |
| a SPIDER checkout | G1 assets + the solve venv it builds | **setup only** |

Everything else `studio setup` creates. There are no sibling-repo paths at run
time — set two paths in `config.yml` and the repo is self-contained.

Three venvs, because each third-party stack pins its own torch:

```
.venv          studio: CLI + recon + verdicts   (pyyaml, numpy, mujoco)  <- uv sync
.venv-kimodo   the demo runtime                 (torch + kimodo)       } studio
.venv-solve    the solve runtime                (torch + warp + SPIDER)} setup
```

`uv sync` alone gets you authoring and reconstruction. The multi-GB CUDA stacks
only arrive when you ask for them.

## Setup (once)

```bash
cp config.example.yml config.yml   # edit kimodo_repo and spider
uv sync                            # studio's own venv
uv run studio setup                # G1 assets + both heavy venvs
```

Stages can be run alone: `studio setup --assets`, `--kimodo`, `--solve`.
The first `studio demo` then downloads `Kimodo-G1-RP-v1` from HuggingFace and
builds the viser web client — a few minutes, once.

Nothing to author yet? This needs no demo and no cluster:

```bash
cp samples/box_carrying.npz raw_motion/
uv run studio recon raw_motion/box_carrying.npz --name demo
uv run studio solve demo            # or: studio panel, to tune it first
```

## Daily loop

```bash
# on the cluster, once per session:
sbatch remote/encoder_euler.sh          # job log prints "encoder node: <node>"

# terminal 1 — tunnel (stays open):
uv run studio tunnel <node>

# terminal 2 — demo:
uv run studio demo                      # http://127.0.0.1:7860

# terminal 3 — pipeline:
uv run studio watch
```

In the browser: write a prompt (and optional timeline constraints), generate,
then **click the sample you like** — with `num_samples > 1` the save buttons
stay disabled until you commit a sample by clicking it. Then either:

- **Verify in place** — the demo's **"Scene recon"** folder: set a run name and
  scene params, click **Reconstruct scene**, and the reconstructed
  table/terrain/box overlays the playing motion right in the demo viewer. It
  writes `raw_motion/<name>.npz` and `runs/<name>/`, ready for the solve panel.
- **Save Example** — the watcher flow: `studio watch` picks it up, reconstructs,
  solves, and prints the LIFT / DynaRetarget verdict.

Then tune and solve:

```bash
uv run studio panel     # :8082 — pick a clip from raw_motion/ (previews on
                        # select), set the contact window against the ghost
                        # box, Reconstruct, tune SBMPC hyperparams, Solve,
                        # and watch reference vs solution side by side
```

## Commands

| command | what it does |
|---|---|
| `studio setup [--assets\|--kimodo\|--solve]` | G1 assets and the two heavy venvs |
| `studio tunnel <node>` | SSH tunnel to the remote text encoder |
| `studio demo [--offline]` | launch the demo + scene-recon add-on |
| `studio watch` | auto-process new Save Example dirs |
| `studio run <dir\|npz>` | full pipeline for one example or motion NPZ |
| `studio recon <dir\|npz>` | reconstruction only; `--box-mass`, `--pick-frame`, … |
| `studio solve <name>` | solve an already-reconstructed run; `--num-samples`, … |
| `studio panel [--run N]` | interactive reconstruct + solve (`:8082`) |
| `studio view <name>` | reference vs. result — opens the panel on that run |
| `studio list` | runs and verdicts |
| `studio promote <name> --to <dir>` | copy passing trials into a central dataset |

## What a run contains

Each run is self-contained under `runs/<name>/`: shimmed inputs, `outputs/`
(scene.xml, reference and solved trajectories, the resolved `solve_config.json`),
`logs/`, and `manifest.json` — prompt, seed, git SHAs, scene and solve params,
and the verdict.

## Repo layout

```
src/studio/
  cli.py             argparse surface; one cmd_* per subcommand
  config.py          config.yml -> Config; scene + solve defaults; run paths
  pipeline.py        reconstruct -> solve -> eval, for one clip
  shim.py            demo save formats -> the batch layout recon expects
  manifest.py        per-run provenance
  watch.py           poll the demo's Save Example dir
  viz.py             reconstruction -> viser handles (viser + trimesh, so
                     only the scripts below may import it)

  recon/             numpy + mujoco, no GPU — runs in studio's venv
    assets.py        G1 meshes/URDF + the scene template
    signal.py        the smoothing every stage shares
    loader.py        Kimodo NPZ -> MuJoCo G1 qpos
    grasp.py         grasp detection + the contact-window override
    graph.py         scene-interaction graph (key links x scene nodes)
    scene.py         box + terrain reconstruction, trial emission

  solve/
    evaluate.py      LIFT / DR verdicts — numpy only, studio's venv
    spider_cfg.py    the resolved solve config (this is what replaced hydra)
    rewards.py       studio's reward stack over SPIDER's
    loop.py          the receding-horizon SBMPC loop   } .venv-solve only
assets/
  scene_template.xml the MuJoCo scene surgery operates on (checked in)
  g1/                meshes + URDF (fetched by `studio setup`, gitignored)

scripts/             run inside the OTHER venvs, never studio's
  panel_app.py         reconstruct + solve panel        (solve venv)
  demo_scene_addon.py  demo + scene-recon overlay       (kimodo venv)
  offline_encoder.py   stand-in text encoder            (kimodo venv)
  encode_via_api.py    snapshot embeddings via tunnel   (kimodo venv)
```

```bash
uv run pytest        # 81 tests: shim, config, manifest, recon, assets,
                     # solve boundary, verdicts
```

One place per concern: scene params in `config.SCENE_DEFAULTS`, solve params in
`config.SOLVE_DEFAULTS`, scene rendering in `viz.py`, the whole solve config in
`solve/spider_cfg.py`.

## How the solve works

Three nested loops (`solve/loop.py`), at the defaults in `SOLVE_DEFAULTS`:

- **outer — receding horizon.** Optimize over a 1.0 s reference window, commit
  only `ctrl_steps` (6 sim steps at 60 Hz), shift the buffer, repeat.
- **middle — annealing.** 16 optimizer passes per control tick, exploration
  noise decaying as `beta_traj ** i`.
- **inner — one MPPI update.** Perturb ~10 spline knots, roll out 2048
  mujoco_warp worlds in parallel over 60 steps, softmax-weight the **top 10%**
  by reward, average into the new nominal control.

The reward is SPIDER's weighted qpos tracking plus two terms studio owns
(`solve/rewards.py`): a **per-block weighted velocity** term, because SPIDER's
flat L2 lets 35 robot dims swamp 6 object dims; and a **simulated-box face
contact** term, because SPIDER pulls palms toward points baked from the
*reference* box pose, which fights the physics once the real box drifts.

## Provenance and licensing

`recon/` is a port of mppi_locoma's reconstruction stack, verified bit-identical
to it (scene.xml, task_info.json, all seven trajectory arrays). The method is
SceneBot's hindsight scene reconstruction (arXiv 2606.27581, Alg. 1).
`solve/loop.py` derives from SPIDER's `examples/run_mjwp.py`, trimmed to the one
configuration studio uses; `solve/rewards.py` and `solve/evaluate.py` come from
mppi_locoma, the latter implementing DynaRetarget's criteria (arXiv 2602.06827).

**SPIDER is CC-BY-NC** (Meta Platforms), and `assets/scene_template.xml` is
derived from it, so this repo inherits that non-commercial restriction. The G1
meshes and URDF are Unitree assets, copied out of a SPIDER install at setup
rather than redistributed here; their `LICENSE` lands in `assets/g1/`.

## Notes

**The contact window is the main knob.** Reconstruction hinges on the pick →
release period: it sets the box rest pose, the box width, and the frames the
solver rewards palm contact. It **defaults to the full clip**. The panel also
runs detection on preview and reports what it found, applied with the
**auto window** button. The orange **ghost box** rides the hands through the
current window, showing the exact object trajectory it implies — edit
`contact start`/`contact end` until the ghost looks right, then Reconstruct.
In the demo, `-1` means full clip.

**Clips that start mid-hold** (hands already on the box at frame 0) build as
*starts-holding* scenes whenever the window starts at frame 0 — the default
covers this. The box spawns in the hands and no support is built underneath.
`--allow-held-start` (and the panel checkbox) additionally lets the
auto-detector recognize held starts on its own. Note the LIFT verdict measures
box final-minus-initial height, so a level carry reads "no" — judge those by
the DR column.

**The solve is not reproducible run-to-run.** Same clip, same config, same
seed: two solves differ by ~0.02 m in final box position (mujoco_warp reductions
on the GPU are not deterministic). Reconstruction *is* exact — it is CPU numpy +
MuJoCo. So judge a solve change by the verdict and by several runs, never by
diffing one trajectory against another.

**Reach-only clips are SKIPPED.** Reconstruction infers the object from
pick/carry hand kinematics; a motion with no object interaction has no scene to
build. That's a verdict, not a bug.

**Embeddings are not persisted by headless generation.** The demo caches prompt
embeddings in `~/.cache/kimodo_demo/embeddings`, but kimodo's
`batch_generate.py` fetches them transiently. While a tunnel is up, run
`.venv-kimodo/bin/python scripts/encode_via_api.py --batch_dir <dir>` once — it
writes `text_embeddings.npz` (auto-detected by `batch_generate.py`) and drops a
copy into `offline_cache/snapshots/`. That prompt then never needs the cluster
again, for any seed, duration, sample count, or constraint set.

**`studio demo --offline` needs no cluster at all.** A local stand-in encoder
(`scripts/offline_encoder.py`, same gradio protocol) serves embeddings from
`offline_cache/snapshots/*.npz` plus anything the demo cached in past tunnel
sessions. Constraints, duration, seeds, and sample count stay freely editable —
only the prompt *text* needs a stored embedding. An unknown prompt generates
text-ignoring motion and prints a loud warning in the server terminal. The
offline session uses an isolated cache dir, so its zero-embeddings can never
pollute the real demo cache.

**Bring the tunnel up before `studio demo`** — at startup the demo prewarms
embeddings for all example prompts. `studio demo` pre-checks the tunnel and
tells you what to start if it's down.

**VRAM**: the demo and the solve share the 16 GB GPU. Reconstruction is CPU-only,
so it costs nothing next to the demo; if the *solve* OOMs while the demo is
open, lower `num_samples` (`studio solve --num-samples 1024`, the panel slider,
or the `solve:` section of `config.yml`).

**The demo's Save Example dir is hardcoded inside the kimodo package**
(`kimodo/assets/demo/examples/kimodo-g1-rp/`). `studio watch` watches it in
place, and saved examples reappear in the demo's Examples dropdown for
re-loading and editing.
