#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=encoder-%j.out

# ─────────────────────────────────────────────────────────────
# LLM2Vec-8B text-encoder server for interactive kimodo_studio use.
#
#   sbatch remote/encoder_euler.sh          # on the cluster
#   tail the job log for "encoder node: <node>"
#   studio tunnel <node>                    # on the workstation
#   studio demo
#
# Paths are per-account; export them before sbatch (or edit the
# defaults below) — the job log lands in the submit directory:
#
#   export KIMODO_SIF=$SCRATCH/kimodo/kimodo.sif
#   export KIMODO_WORKSPACE=$HOME/kimodo_ws/kimodo
#   export KIMODO_HF_CACHE=$SCRATCH/kimodo/hf_cache
#   sbatch --export=ALL remote/encoder_euler.sh
#
# The server has NO auth, so it binds 127.0.0.1 and is reachable
# only through the SSH tunnel. CUDA_VISIBLE_DEVICES=0 avoids
# llm2vec's multi-GPU spawn branch. 4x8192 MB RAM because the PEFT
# merge into the 16 GB bf16 base model happens on CPU.
# --time bounds the authoring session; raise it for longer ones.
# PORT must match `encoder.port` in the workstation's config.yml.
# ─────────────────────────────────────────────────────────────

set -euo pipefail
module load eth_proxy

SIF="${KIMODO_SIF:-${SCRATCH:-$HOME}/kimodo/kimodo.sif}"
WORKSPACE="${KIMODO_WORKSPACE:-$HOME/kimodo_ws/kimodo}"
HF_CACHE="${KIMODO_HF_CACHE:-${SCRATCH:-$HOME}/kimodo/hf_cache}"
PORT="${KIMODO_ENCODER_PORT:-9550}"

for var in SIF WORKSPACE HF_CACHE; do
  if [ ! -e "${!var}" ]; then
    echo "missing $var: ${!var}   (export KIMODO_${var}=... before sbatch)" >&2
    exit 1
  fi
done

echo "encoder node: $(hostname)"

apptainer exec \
  --nv \
  --containall \
  --cleanenv \
  --writable-tmpfs \
  --env HF_HOME=/workspace/.cache/huggingface \
  --env PYTHONPATH=/workspace \
  --env HF_HUB_OFFLINE=1 \
  --env GRADIO_SERVER_NAME=127.0.0.1 \
  --env GRADIO_SERVER_PORT="$PORT" \
  --env CUDA_VISIBLE_DEVICES=0 \
  -B "$HF_CACHE":/workspace/.cache/huggingface:rw \
  -B "$WORKSPACE":/workspace:rw \
  "$SIF" \
  python3 -m kimodo.scripts.run_text_encoder_server
