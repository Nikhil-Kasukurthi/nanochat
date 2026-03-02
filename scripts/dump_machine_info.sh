#!/bin/bash
# dump_machine_info.sh — Capture machine topology for GPU training diagnostics
# Run inside the container on the GPU node. Output goes to stdout.
# Usage: bash dump_machine_info.sh > machine_info.txt 2>&1

set -euo pipefail

divider() { echo -e "\n===== $1 ====="; }

divider "HOSTNAME & DATE"
hostname
date -u +"%Y-%m-%dT%H:%M:%SZ"

divider "OS / KERNEL"
uname -a
cat /etc/os-release 2>/dev/null | head -5 || true

# ── CPU ──────────────────────────────────────────
divider "LSCPU"
lscpu

divider "NPROC (cgroup-effective)"
nproc

divider "CGROUP CPU LIMITS"
# cgroup v2
if [ -f /sys/fs/cgroup/cpu.max ]; then
    echo "cgroup v2: $(cat /sys/fs/cgroup/cpu.max)"
# cgroup v1
elif [ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    echo "quota_us: $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)"
    echo "period_us: $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)"
    effective_cpus=$(awk '{printf "%.1f", $1/$2}' <(paste /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us) 2>/dev/null || true)
    [ -n "$effective_cpus" ] && echo "effective CPUs: $effective_cpus"
else
    echo "No cgroup CPU limits found (unrestricted or unknown cgroup layout)"
fi

divider "CPUSET (allowed CPUs)"
if [ -f /sys/fs/cgroup/cpuset/cpuset.cpus ]; then
    echo "cgroup v1 cpuset: $(cat /sys/fs/cgroup/cpuset/cpuset.cpus)"
elif [ -f /sys/fs/cgroup/cpuset.cpus.effective ]; then
    echo "cgroup v2 cpuset: $(cat /sys/fs/cgroup/cpuset.cpus.effective)"
else
    # fallback: read from proc
    grep -i "cpus_allowed_list" /proc/self/status 2>/dev/null || echo "unknown"
fi

divider "NUMA TOPOLOGY"
numactl --hardware 2>/dev/null || echo "numactl not available"

# ── MEMORY ───────────────────────────────────────
divider "MEMORY"
free -h

divider "CGROUP MEMORY LIMIT"
if [ -f /sys/fs/cgroup/memory.max ]; then
    echo "cgroup v2: $(cat /sys/fs/cgroup/memory.max)"
elif [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    echo "limit_in_bytes: $limit ($(awk "BEGIN{printf \"%.1f GB\", $limit/1073741824}"))"
else
    echo "No cgroup memory limits found"
fi

# ── GPU ──────────────────────────────────────────
divider "NVIDIA-SMI"
nvidia-smi

divider "GPU TOPOLOGY (nvidia-smi topo -m)"
nvidia-smi topo -m

divider "NVLINK STATUS"
nvidia-smi nvlink --status 2>/dev/null || echo "nvlink query not supported"

divider "GPU CLOCKS & POWER"
nvidia-smi --query-gpu=index,name,clocks.current.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu --format=csv,noheader

# ── NCCL / NETWORKING ────────────────────────────
divider "NCCL ENV VARS"
env | grep -i nccl | sort || echo "No NCCL env vars set"

divider "CUDA ENV VARS"
env | grep -i cuda | sort || echo "No CUDA env vars set"

divider "SHARED MEMORY (/dev/shm)"
df -h /dev/shm 2>/dev/null || echo "/dev/shm not mounted"

divider "NETWORK INTERFACES"
ip -br addr 2>/dev/null || ifconfig 2>/dev/null || echo "No network tools available"

divider "INFINIBAND / RDMA"
ibstat 2>/dev/null || ibv_devinfo 2>/dev/null || echo "No IB tools available"

# ── SOFTWARE ─────────────────────────────────────
divider "PYTHON & TORCH"
python3 -c "
import torch, sys
print(f'Python {sys.version}')
print(f'PyTorch {torch.__version__}')
print(f'CUDA {torch.version.cuda}')
print(f'cuDNN {torch.backends.cudnn.version()}')
print(f'NCCL {torch.cuda.nccl.version()}')
print(f'GPUs visible: {torch.cuda.device_count()}')
" 2>/dev/null || echo "torch not available"

divider "NCCL LIBRARY VERSION"
ldconfig -p 2>/dev/null | grep nccl || echo "NCCL library not in ldconfig"

echo -e "\n===== DONE ====="
