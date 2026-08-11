# kimodo_studio

Generate a single G1 motion clip with
[Kimodo](https://github.com/nv-tlabs/kimodo), reconstruct the scene it implies,
and solve for a **dynamically feasible robot+object trajectory**
([SPIDER](https://github.com/facebookresearch/spider) + mujoco_warp).

<!-- TODO: demo gif — panel side-by-side (reference vs. solved) is the money shot -->
<!-- ![demo](docs/demo.gif) -->

## Setup (once)

Everything runs on your workstation (one CUDA GPU, ~16 GB — the demo and the
solve share it) **except the text encoder**: prompts are embedded by
LLM2Vec-8B, too heavy for a desktop GPU, so it runs as a cluster job (the
provided script requests an RTX 3090) and streams back over an SSH tunnel.

**On the cluster** — stage three things, once:

```bash
# 1. apptainer image, built from kimodo's Dockerfile (see extern/kimodo):
#    docker build -t kimodo extern/kimodo && apptainer build kimodo.sif docker-daemon://kimodo:latest
scp kimodo.sif euler:'$SCRATCH'/kimodo/kimodo.sif
# 2. a kimodo checkout:
ssh euler 'git clone https://github.com/nv-tlabs/kimodo.git $HOME/kimodo_ws/kimodo'
# 3. the encoder weights (the job runs offline):
ssh euler 'module load eth_proxy && export HF_HOME=$SCRATCH/kimodo/hf_cache && \
  huggingface-cli download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp && \
  huggingface-cli download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised'
```

Non-default locations: export `KIMODO_SIF` / `KIMODO_WORKSPACE` /
`KIMODO_HF_CACHE`, submit with `sbatch --export=ALL`.

**Locally:**

```bash
git clone <this-repo> && cd kimodo_studio
uv sync              # studio's own venv (CPU-only recon + verdicts)
uv run studio setup  # submodules + G1 assets + both heavy venvs (~13 GB)
uv run pytest        # verify
```

No config file needed — the upstreams are pinned submodules under `extern/`
and `studio setup` fetches them itself. `config.yml`
(see `config.example.yml`) only overrides defaults, e.g. to use your own
kimodo fork.

## Workflow

**1. Text embeddings** — once per authoring session:

```bash
sbatch remote/encoder_euler.sh   # on the cluster; job log prints "encoder node: <node>"
uv run studio tunnel <node>      # keep open
```

No cluster today? `studio demo --offline` serves prompts banked in
`offline_cache/`. Bank new ones while a tunnel is up:
`.venv-kimodo/bin/python scripts/encode_via_api.py --file offline_cache/prompts.txt --split`

**2. Motion generation** — the Kimodo viser demo
([how to use it](https://research.nvidia.com/labs/sil/projects/kimodo/)),
wrapped with a scene-recon add-on:

```bash
uv run studio demo               # http://127.0.0.1:7860
uv run studio watch              # optional: auto recon+solve+verdict on every Save Example
```

Prompt → **Generate** → **click the sample you like**, then **Save Example**
(picked up by `studio watch`) or the add-on's **Reconstruct scene** (overlays
the inferred scene in the demo viewer).

**3. Tune and solve:**

```bash
uv run studio panel     # :8082 — pick a clip from raw_motion/ (previews on
                        # select), set the contact window against the ghost
                        # box, Reconstruct, tune SBMPC hyperparams, Solve,
                        # and watch reference vs solution side by side
```

Or scripted, no GUI — works on the checked-in sample clip with no demo and no
cluster:

```bash
cp samples/box_carrying.npz raw_motion/
uv run studio recon raw_motion/box_carrying.npz --name demo
uv run studio solve demo         # --num-samples 1024 if VRAM is tight
uv run studio list               # runs + LIFT / DR verdicts
```

`studio -h` lists the rest (`run`, `view`, `promote`, per-flag help on each).

## Notes

- **The contact window is the main knob**: pick → release sets the box size,
  rest pose, and contact rewards. Default is the full clip; in the panel, drag
  `contact start`/`end` until the orange ghost box looks right.
- **Solves are not run-to-run reproducible** (GPU reductions) — judge by the
  verdict over several runs. Reconstruction is exact.
- **Reach-only clips are SKIPPED**: no object interaction, no scene to build.
- Runs are self-contained under `runs/<name>/` (scene.xml, trajectories,
  `manifest.json`: prompt, seed, params, verdict). Internals live in
  `src/studio/recon/` and `src/studio/solve/`.

## References

- **Kimodo** — Rempe et al., *Kimodo: Scaling Controllable Human Motion
  Generation* — [arXiv:2603.15546](https://arxiv.org/abs/2603.15546) ·
  [project page](https://research.nvidia.com/labs/sil/projects/kimodo/).
  The motion generator this studio authors with.
- **SPIDER** — Pan et al., *SPIDER: Scalable Physics-Informed Dexterous
  Retargeting* — [arXiv:2511.09484](https://arxiv.org/abs/2511.09484) ·
  [code](https://github.com/facebookresearch/spider). Supplies the sampling
  optimizer and batched mujoco_warp simulator; `solve/loop.py` derives from
  its `run_mjwp.py`.
- **SceneBot** — *SceneBot: Contact-Prompted General Humanoid Whole Body
  Tracking with Scene-Interaction* —
  [arXiv:2606.27581](https://arxiv.org/abs/2606.27581) ·
  [project page](https://ericcsr.github.io/scenebot/). `recon/` implements
  its hindsight scene reconstruction (Alg. 1).
- **DynaRetarget** — Dhedin et al., *DynaRetarget: Dynamically-Feasible
  Retargeting using Sampling-Based Trajectory Optimization* —
  [arXiv:2602.06827](https://arxiv.org/abs/2602.06827). The LIFT / DR
  verdicts implement its feasibility criteria.

## Licensing

**SPIDER is CC-BY-NC** (Meta Platforms) and `assets/scene_template.xml`
derives from it, so this repo inherits the non-commercial restriction; SPIDER
is referenced as a submodule pointer, not redistributed. The G1 meshes/URDF
are Unitree assets, copied from the SPIDER checkout at setup; their license
lands in `assets/g1/LICENSE`.
