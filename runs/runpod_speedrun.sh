#!/bin/bash
set -e

# Train a nanochat LLM on 8xB200 GPUs with NVFP4 precision.
# Uses TransformerEngine for FP4 and Flash Attention 4 for Blackwell.
#
# Example launch:
#   screen -L -Logfile speedrun.log -S speedrun bash runs/runpod_speedrun.sh
# With wandb:
#   WANDB_RUN=d24 screen -L -Logfile speedrun.log -S speedrun bash runs/runpod_speedrun.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/workspace/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# System dependencies
apt-get update -qq && apt-get install -y -qq python3-dev vim tmux 2>/dev/null

# -----------------------------------------------------------------------------
# Python venv + dependencies

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi
export PATH="$HOME/.local/bin:$PATH"

[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# TransformerEngine for NVFP4 training
# Building from source needs CUDA headers — nvidia pip packages have them but not on default paths.
# LD_LIBRARY_PATH is needed at runtime for libcublas/libcudnn.]
uv pip install -U torch
echo "Installing TransformerEngine for NVFP4 training..."
NVIDIA_DIRS=$(find .venv/lib/python*/site-packages/nvidia /venv/main/lib/python*/site-packages/nvidia -maxdepth 2 \( -name 'include' -o -name 'lib' \) -type d 2>/dev/null)
NVIDIA_INCLUDES=$(echo "$NVIDIA_DIRS" | grep include | tr '\n' ':')
NVIDIA_LIBS=$(echo "$NVIDIA_DIRS" | grep '/lib$' | tr '\n' ':')
CPATH="${NVIDIA_INCLUDES}${CPATH}" LIBRARY_PATH="${NVIDIA_LIBS}${LIBRARY_PATH}" \
    uv pip install --no-build-isolation "transformer_engine[pytorch]"
# Set runtime library path (re-scan after install since TE may add nvidia packages)
NVIDIA_LIBS=$(find .venv/lib/python*/site-packages/nvidia /venv/main/lib/python*/site-packages/nvidia -maxdepth 2 -name 'lib' -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH}"

# Flash Attention 4 for Blackwell
echo "Installing flash-attn-4 for Blackwell..."
uv pip install "flash-attn-4[cu13]" --prerelease=allow
uv pip install "nvidia-cutlass-dsl==4.4.2" "nvidia-cutlass-dsl-libs-base==4.4.2" "nvidia-cutlass-dsl-libs-cu13==4.4.2"

# -----------------------------------------------------------------------------
# wandb (optional)
WANDB_RUN=${WANDB_RUN:-dummy}

# -----------------------------------------------------------------------------
# Report header
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 350 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp4 --run=$WANDB_RUN
# evaluate the model: CORE metric, BPB on train/val, and draw samples
python -m scripts.base_eval --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT

curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT and eval the model
python -m scripts.chat_sft -- --device-batch-size=32 --total-batch-size=32768 --run=$WANDB_RUN
python -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# Report
python -m nanochat.report generate
