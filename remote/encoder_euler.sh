#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=/cluster/home/yixili/kimodo_ws/artifacts/encoder-%j.out

# ─────────────────────────────────────────────────────────────
# LLM2Vec-8B text-encoder server for interactive kimodo_studio use.
#
#   sbatch remote/encoder_euler.sh          # on the cluster
#   tail the job log for "encoder node: <node>"
#   studio tunnel <node>                    # on the workstation
#   studio demo
#
# The server has NO auth, so it binds 127.0.0.1 and is reachable
# only through the SSH tunnel. CUDA_VISIBLE_DEVICES=0 avoids
# llm2vec's multi-GPU spawn branch. 4x8192 MB RAM because the PEFT
# merge into the 16 GB bf16 base model happens on CPU.
# --time bounds the authoring session; raise it for longer ones.
# ─────────────────────────────────────────────────────────────

set -e
module load eth_proxy

SIF=/cluster/scratch/yixili/kimodo/kimodo.sif
WORKSPACE=/cluster/home/yixili/kimodo_ws/kimodo
HF_CACHE=/cluster/scratch/yixili/kimodo/hf_cache

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
  --env GRADIO_SERVER_PORT=9550 \
  --env CUDA_VISIBLE_DEVICES=0 \
  -B $HF_CACHE:/workspace/.cache/huggingface:rw \
  -B $WORKSPACE:/workspace:rw \
  $SIF \
  python3 -m kimodo.scripts.run_text_encoder_server
