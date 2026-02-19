#!/bin/bash
set -e

# Profile communication overhead for the d26 model (GPT-2 performance) on 8xH100.
# Run this on PCIe, SXM, and NVL instances to compare interconnect overhead.
# Takes ~5 minutes per run (10 profiled steps + 3 warmup).
#
# 1) Example launch (simplest):
# bash runs/runpod_profile_comms.sh
# 2) Example launch with nsys tracing:
# NSYS=1 bash runs/runpod_profile_comms.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/workspace/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# System dependencies

apt-get update -qq && apt-get install -y -qq python3-dev vim tmux 2>/dev/null

# Install nsys if not present
if ! command -v nsys &> /dev/null; then
    apt-get install -y -qq nsight-systems-cli 2>/dev/null || {
        apt-get install -y --no-install-recommends gnupg 2>/dev/null
        echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" \
            | tee /etc/apt/sources.list.d/nvidia-devtools.list
        apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub 2>/dev/null
        apt-get update -qq && apt-get install -y -qq nsight-systems-cli
    }
fi

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env; }
export PATH="$HOME/.local/bin:$PATH"
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Data + tokenizer (profiling only needs a few shards)

python -m nanochat.dataset -n 8
# train tokenizer if not already present
[ -f "$NANOCHAT_BASE_DIR/tok65536.model" ] || python -m scripts.tok_train

# -----------------------------------------------------------------------------
# Profile the d26 model on all available GPUs
# d26 is the model that reaches GPT-2 performance in the full training run

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NSYS=${NSYS:-0}

PROFILE_CMD="torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.profile_comms \
    -- --depth 26 --num-steps 10 --warmup-steps 3 --device-batch-size 32 --output-dir profile_output"

if [ "$NSYS" = "1" ]; then
    nsys profile \
        --python-backtrace=cuda \
        --pytorch autograd-shapes-nvtx \
        -o profile_output/nsys_trace \
        --trace=cuda,nvtx,osrt \
        --capture-range=cudaProfilerApi \
        $PROFILE_CMD
else
    $PROFILE_CMD
fi
