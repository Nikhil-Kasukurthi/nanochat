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
echo "Checking NCCL communication across $NUM_GPUS GPUs..."

NCCL_CHECK=$(mktemp /tmp/nccl_check_XXXXXX.py)
cat > "$NCCL_CHECK" << 'PYEOF'
import torch, torch.distributed as dist, os, sys
rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
device = torch.device('cuda', rank)
try:
    dist.init_process_group(backend='nccl', device_id=device)
    dist.barrier()
    t = torch.ones(1024, device=device) * (rank + 1)
    dist.all_reduce(t)
    expected = sum(range(1, dist.get_world_size() + 1)) * 1024
    assert abs(t.sum().item() - expected) < 1e-3, f'all_reduce mismatch: {t.sum().item()} != {expected}'
    if rank == 0:
        gpu_name = torch.cuda.get_device_name(0)
        nccl_ver = '.'.join(str(x) for x in torch.cuda.nccl.version())
        print(f'NCCL OK: {dist.get_world_size()} GPUs communicating ({gpu_name}, NCCL {nccl_ver})')
    dist.destroy_process_group()
except Exception as e:
    if rank == 0:
        print(f'\nNCCL HEALTH CHECK FAILED: {e}', file=sys.stderr)
        print(file=sys.stderr)
        print('This pod has broken NCCL communication. Common causes:', file=sys.stderr)
        print('  1. NCCL SHM bug -- corrupt /dev/shm/nccl-* segments (seen on some NVL containers)', file=sys.stderr)
        print('  2. NCCL version mismatch with driver/CUDA', file=sys.stderr)
        print('  3. GPU topology not properly exposed to container', file=sys.stderr)
        print(file=sys.stderr)
        print('Quick workaround:  NCCL_SHM_DISABLE=1 bash runs/runpod_profile_comms.sh', file=sys.stderr)
        print('  (WARNING: disabling SHM gives invalid perf numbers on NVL nodes)', file=sys.stderr)
        print(file=sys.stderr)
        print('Recommended: terminate this pod and try a different one.', file=sys.stderr)
    sys.exit(1)
PYEOF

timeout 60 torchrun --standalone --nproc_per_node=$NUM_GPUS "$NCCL_CHECK" \
    || { rm -f "$NCCL_CHECK"; echo "FATAL: NCCL check failed. Abort this pod and try another."; exit 1; }
rm -f "$NCCL_CHECK"

# -----------------------------------------------------------------------------
# System dependencies

apt-get update -qq && apt-get install -y -qq python3-dev vim tmux rsync 2>/dev/null

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
# -----------------------------------------------------------------------------
# Profile d12 and d26 models on all available GPUs
# d12 is the smallest standard model; d26 reaches GPT-2 performance

# --- d12 profiling ---

PROFILE_CMD_D12="torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.profile_comms \
    -- --depth 12 --num-steps 10 --warmup-steps 3 --device-batch-size 32 --fp8 --output-dir profile_output/d12"
mkdir -p profile_output/d12/nsys_trace
nsys profile \
      --python-backtrace=cuda \
      --pytorch autograd-shapes-nvtx \
      -o profile_output/d12/nsys_trace \
      --trace=cuda,nvtx,osrt \
      --capture-range=cudaProfilerApi \
      $PROFILE_CMD_D12

# --- d26 profiling ---

PROFILE_CMD_D26="torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.profile_comms \
    -- --depth 26 --num-steps 10 --warmup-steps 3 --device-batch-size 16 --fp8 --output-dir profile_output/d26"
mkdir -p profile_output/d26/nsys_trace
nsys profile \
      --python-backtrace=cuda \
      --pytorch autograd-shapes-nvtx \
      -o profile_output/d26/nsys_trace \
      --trace=cuda,nvtx,osrt \
      --capture-range=cudaProfilerApi \
      $PROFILE_CMD_D26
