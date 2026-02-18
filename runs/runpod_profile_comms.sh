#!/bin/bash
set -e

# Profile communication overhead on RunPod multi-GPU instances.
# Measures all-reduce / reduce-scatter / all-gather time in DistMuonAdamW
# across different GPU interconnects (PCIe, NVLink SXM, NVL72).
#
# RunPod-specific:
# - Installs NVIDIA Nsight Systems (nsys) for kernel-level tracing
# - Installs python3-dev (needed for torchao FP8 CUDA extension JIT)
# - Adds ~/.local/bin to PATH (where uv installs on RunPod)
#
# Usage:
#   bash runs/runpod_profile_comms.sh                     # PyTorch Profiler only
#   NSYS=1 bash runs/runpod_profile_comms.sh              # also capture Nsight traces
#   DEPTH=20 NUM_GPUS=4 bash runs/runpod_profile_comms.sh # override model/gpu count

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
DEPTH=${DEPTH:-12}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NUM_STEPS=${NUM_STEPS:-10}
WARMUP_STEPS=${WARMUP_STEPS:-3}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-32}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
OUTPUT_DIR=${OUTPUT_DIR:-"profile_output"}
NSYS=${NSYS:-0}  # set NSYS=1 to wrap with nsys profile

echo "=== nanochat Communication Profiler (RunPod) ==="
echo "Config: depth=$DEPTH gpus=$NUM_GPUS steps=$NUM_STEPS warmup=$WARMUP_STEPS"
echo "        device_batch_size=$DEVICE_BATCH_SIZE seq_len=$MAX_SEQ_LEN"
echo "        output_dir=$OUTPUT_DIR nsys=$NSYS"

# -----------------------------------------------------------------------------
# Environment
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/workspace/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# System dependencies
echo "Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq python3-dev vim tmux 2>/dev/null

# Install NVIDIA Nsight Systems (nsys) for kernel-level GPU profiling
if ! command -v nsys &> /dev/null; then
    echo "Installing NVIDIA Nsight Systems..."
    apt-get install -y -qq nsight-systems-cli 2>/dev/null || {
        echo "Trying alternative nsys installation..."
        apt-get install -y --no-install-recommends gnupg 2>/dev/null
        echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" \
            | tee /etc/apt/sources.list.d/nvidia-devtools.list
        apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub 2>/dev/null
        apt-get update -qq
        apt-get install -y -qq nsight-systems-cli
    }
    echo "nsys installed: $(nsys --version 2>/dev/null || echo 'installation may have failed')"
else
    echo "nsys already installed: $(nsys --version 2>/dev/null)"
fi

# -----------------------------------------------------------------------------
# Python venv setup with uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi
export PATH="$HOME/.local/bin:$PATH"

[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Download minimal data (profiling only needs a few batches)
# 8 shards = ~2B chars, enough for the profiling loop to have data to chew on
echo "Ensuring training data is available..."
python -m nanochat.dataset -n 8

# Train tokenizer if not already present (needed to init the model)
TOKENIZER_PATH="$NANOCHAT_BASE_DIR/tok65536.model"
if [ ! -f "$TOKENIZER_PATH" ]; then
    echo "Training tokenizer (first run only)..."
    python -m scripts.tok_train
fi

# -----------------------------------------------------------------------------
# Run profiling
echo ""
echo "=== Starting communication profiling ==="
echo "Model: d${DEPTH} | GPUs: ${NUM_GPUS} | Steps: ${WARMUP_STEPS} warmup + ${NUM_STEPS} profiled"
echo ""

PROFILE_CMD="torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.profile_comms \
    -- \
    --depth $DEPTH \
    --num-steps $NUM_STEPS \
    --warmup-steps $WARMUP_STEPS \
    --device-batch-size $DEVICE_BATCH_SIZE \
    --max-seq-len $MAX_SEQ_LEN \
    --output-dir $OUTPUT_DIR"

if [ "$NSYS" = "1" ]; then
    # Wrap with Nsight Systems — captures NVTX markers, CUDA kernels, NCCL comms
    NSYS_OUTPUT="${OUTPUT_DIR}/nsys_trace"
    echo "Running under nsys profile (output: ${NSYS_OUTPUT}.nsys-rep)"
    nsys profile \
        -o "$NSYS_OUTPUT" \
        --trace=cuda,nvtx,osrt \
        --capture-range=cudaProfilerApi \
        --stop-on-range-end=true \
        $PROFILE_CMD
else
    # Standard run — PyTorch Profiler traces + summary
    # NVTX markers are present but dormant (near-zero overhead without nsys)
    $PROFILE_CMD
fi

echo ""
echo "=== Profiling complete ==="
echo "Results in: $OUTPUT_DIR/"
echo "  profile_results.json  — machine-readable for cross-instance comparison"
echo "  profile_results.md    — human-readable summary for blog"
if [ "$NSYS" = "1" ]; then
    echo "  nsys_trace.nsys-rep   — open with Nsight Systems GUI or nsys stats"
fi
echo ""
echo "To compare across instances, run on different GPU types and diff the JSON files."
