# motion_studio

Generate a single G1 motion clip with [Kimodo](https://github.com/nv-tlabs/kimodo), reconstruct the scene it implies, and solve for a **dynamically feasible robot+object trajectory**
([SPIDER](https://github.com/facebookresearch/spider) + mujoco_warp).

<!-- regenerate: MUJOCO_GL=egl .venv-solve/bin/python scripts/render_docs_gifs.py -->

**Generate** — Kimodo turns a text prompt into a motion clip (the checked-in
`samples/box_carrying.npz`):

![Kimodo motion generation](docs/generate.gif)

**Reconstruct** — the clip is retargeted to the G1 and the scene it implies
(the box and its placement) is inferred in hindsight:

![hindsight scene reconstruction](docs/recon.gif)

**Solve** — sampling-based MPC in mujoco_warp turns the kinematic reference
into a dynamically feasible robot+object trajectory:

![kinematic reference vs. solved trajectory](docs/solve.gif)

## Setup (once)

Everything runs on your workstation **except the text encoder**: prompts are embedded by
LLM2Vec-8B, too heavy for a desktop GPU, so it runs as a cluster job and streams back over an SSH tunnel.

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

## Downstream tasks

Recon and solve are pluggable per **task** (`src/studio/tasks/`); every run
records its task and `studio solve` / `studio list` follow it. `box_carry`
(everything above) is the default. The second task is collision-free
augmentation:

**`under_table`** — an under-table-pick clip implies a table; place a
randomized one (slab + 4 legs, seeded jitter + yaw) over the ducking robot
from head-trajectory FK, then re-solve with receding-horizon MPPI so the
robot tracks the reference while actually avoiding the now-solid table
(tracking + stability + analytic-SDF avoidance rewards). Verification:
SDF penetration, pelvis drift, and the pick-hand task-preservation check.

```bash
uv run studio recon <under_table_pick clip>.npz --task under_table \
    --name ut0 --scene-seed 3          # placement is seeded: new seed, new scene
uv run studio solve ut0                # --set num_samples=1024 if VRAM is tight
uv run studio list                     # PASS = collision-free + task preserved
```

Or in the panel: `uv run studio panel`, task `under_table`, pick a clip from
`raw_motion/` → **1. Estimate table** (change the seed / difficulty knobs and
re-estimate; the table redraws) → **2. Solve MPPI** → reference (transparent)
vs augmented (solid) playback. `studio view <run>` opens an existing run in
the right mode.

**`kick`** — a Kimodo-generated kick clip; place a randomized
floor-standing box in the kicking foot's approach path (seeded position
along the path, lateral/yaw jitter) so the reference swing penetrates it,
then re-solve. Placement guarantees the kick point itself stays outside
the box and the rest of the body clears it — only the kicking leg
conflicts, and re-pathing it is the solver's work. Verification adds the
kick-preservation check (kicking foot vs the reference kick point).

```bash
uv run studio recon <kick clip>.npz --task kick --name k0 --scene-seed 2
uv run studio solve k0
```

Task knobs (difficulty, reward weights, verify thresholds) live in
`src/studio/tasks/<task>_params.py`; override per machine via config.yml's
`tasks:` section or per run via `--set key=value`. The under-table
estimation and both tasks' rewards port the `mppi_obstacle` experiment's
certified defaults (see that repo's README for the difficulty sweeps);
the kick solve weights are starting points, not yet swept.

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
