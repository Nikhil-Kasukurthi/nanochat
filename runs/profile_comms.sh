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

export NANOCHAT_BASE_DIR="/workspace/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# CUDA sanity check — fail fast if the pod's GPU/driver setup is broken.
# This catches "CUDA unknown error" from mismatched drivers, bad templates, etc.
# before we spend minutes installing dependencies and downloading data.

echo "Checking CUDA availability..."
nvidia-smi || { echo "FATAL: nvidia-smi failed — no GPU driver found."; exit 1; }

DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "NVIDIA driver version: $DRIVER_VER"

python3 -c "
import torch; torch.cuda.init()
print(f'PyTorch {torch.__version__}, built with CUDA {torch.version.cuda}')
n = torch.cuda.device_count()
assert n > 0, 'No CUDA devices found after init'
for i in range(n):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'CUDA OK: {n} device(s) ready')
" || { echo "FATAL: CUDA initialization failed. See diagnostics above."; exit 1; }

# -----------------------------------------------------------------------------
# NCCL sanity check — fail fast if inter-GPU communication is broken.
# This catches NCCL SHM bugs (e.g. corrupt segment names on some NVL containers)
# before we spend minutes installing dependencies and downloading data.
# Uses the base image's torchrun + torch (no venv needed).

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)

# OMP / NCCL tuning — set after GPU count is known
export OMP_NUM_THREADS=$(($(nproc) / NUM_GPUS))
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export NCCL_IB_DISABLE=1        # Vast IB ports are Down; harmless elsewhere
export NCCL_P2P_LEVEL=NVL       # prefer NVLink for peer-to-peer
export NCCL_SOCKET_IFNAME=eth0

echo "Checking NCCL communication across $NUM_GPUS GPUs..."
timeout 60 torchrun --standalone --nproc_per_node=$NUM_GPUS scripts/nccl_check.py \
#    || { echo "FATAL: NCCL check failed. Abort this pod and try another."; exit 1; }

# NUMA diagnostics — log topology for the profiling results
# Actual NUMA pinning (CPU affinity + memory binding) is handled per-rank
# inside Python by numa_pin() in nanochat/common.py — no shell wrapper needed.

NUMA_NODES=$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l)
echo "============================================================"
echo "  NUMA & GPU TOPOLOGY DIAGNOSTICS"
echo "============================================================"
echo ""
echo "--- NUMA hardware ---"
if [ "$NUMA_NODES" -gt 1 ]; then
    echo "NUMA: $NUMA_NODES nodes detected"
    echo "  CPU affinity: torchrun --numa-binding"
    echo "  Memory binding: set_membind() in Python"
    echo ""
    numactl --hardware 2>/dev/null || echo "  numactl not available"
else
    echo "NUMA: single node (or sysfs unavailable) — no pinning needed"
fi

echo ""
echo "--- NUMA node ↔ CPU mapping ---"
for node_dir in /sys/devices/system/node/node[0-9]*; do
    node=$(basename "$node_dir")
    cpulist=$(cat "$node_dir/cpulist" 2>/dev/null || echo "unknown")
    meminfo_total=$(awk '/MemTotal/ {printf "%.1f GB", $4/1048576}' "$node_dir/meminfo" 2>/dev/null || echo "unknown")
    echo "  $node: CPUs [$cpulist]  Memory: $meminfo_total"
done

echo ""
echo "--- GPU topology (nvidia-smi topo -m) ---"
nvidia-smi topo -m 2>/dev/null || echo "  nvidia-smi topo not available"

echo ""
echo "--- GPU ↔ NUMA node mapping (PCI sysfs) ---"
for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
    pci_bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$gpu_idx" 2>/dev/null | tr '[:upper:]' '[:lower:]')
    if [ -n "$pci_bus" ]; then
        # Strip domain prefix (0000:) for sysfs lookup
        pci_short=$(echo "$pci_bus" | sed 's/^0000://')
        numa_node=$(cat "/sys/bus/pci/devices/$pci_short/numa_node" 2>/dev/null \
                 || cat "/sys/bus/pci/devices/$pci_bus/numa_node" 2>/dev/null \
                 || echo "unknown")
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$gpu_idx" 2>/dev/null)
        echo "  GPU $gpu_idx ($gpu_name): PCI $pci_bus → NUMA node $numa_node"
    fi
done

echo ""
echo "--- NVLink status ---"
nvidia-smi nvlink --status 2>/dev/null || echo "  NVLink query not supported"
echo "============================================================"
echo ""

# -----------------------------------------------------------------------------
# Profile d12 and d26 models on all available GPUs
# d12 is the smallest standard model; d26 reaches GPT-2 performance

# -----------------------------------------------------------------------------
# System dependencies

apt-get update -qq && apt-get install -y -qq python3-dev vim tmux rsync numactl 2>/dev/null

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

mkdir -p profile_output

# --- d12 profiling ---

mkdir -p profile_output/d12/nsys_trace
# nsys sends SIGTERM to children during teardown, producing a non-zero exit.
# || true prevents set -e from aborting before d26 profiling runs.
export TORCHRUN="torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.profile_comms --"
nsys profile \
      --python-backtrace=cuda \
      --pytorch autograd-shapes-nvtx \
      -o profile_output/d12/nsys_trace \
      --trace=cuda,nvtx,osrt \
      --capture-range=cudaProfilerApi \
      $TORCHRUN --depth 12 --num-steps 10 --warmup-steps 3 --device-batch-size 32 --fp8 --output-dir profile_output/d12 \
      || true

# --- d26 profiling ---

mkdir -p profile_output/d26/nsys_trace
nsys profile \
      --python-backtrace=cuda \
      --pytorch autograd-shapes-nvtx \
      -o profile_output/d26/nsys_trace \
      --trace=cuda,nvtx,osrt \
      --capture-range=cudaProfilerApi \
      $TORCHRUN --depth 26 --num-steps 10 --warmup-steps 3 --device-batch-size 16 --fp8 --output-dir profile_output/d26 \
      || true
