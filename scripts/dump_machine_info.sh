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

# ── NETWORK DETAILS ─────────────────────────────
divider "LINK SPEEDS"
for iface in $(ls /sys/class/net/ 2>/dev/null); do
    speed=$(cat /sys/class/net/$iface/speed 2>/dev/null || echo "unknown")
    operstate=$(cat /sys/class/net/$iface/operstate 2>/dev/null || echo "unknown")
    echo "$iface: speed=${speed}Mbps state=$operstate"
done

divider "DEFAULT ROUTE & GATEWAY"
ip route show default 2>/dev/null || route -n 2>/dev/null || echo "No routing tools available"

divider "DNS CONFIGURATION"
cat /etc/resolv.conf 2>/dev/null || echo "No resolv.conf found"

divider "DNS LATENCY"
{ time nslookup google.com 2>/dev/null; } 2>&1 | tail -3 || echo "nslookup not available"

divider "PING LATENCY (google.com)"
ping -c 5 -W 2 google.com 2>/dev/null | tail -2 || echo "ping not available or blocked"

divider "TCP TUNING PARAMETERS"
for param in net.core.rmem_max net.core.wmem_max net.ipv4.tcp_rmem net.ipv4.tcp_wmem net.core.netdev_max_backlog net.ipv4.tcp_mtu_probing; do
    val=$(sysctl -n $param 2>/dev/null || echo "unavailable")
    echo "$param = $val"
done

divider "INTERNET BANDWIDTH (curl-based)"
echo "Download test (100MB file from Cloudflare):"
curl -so /dev/null -w "  speed: %{speed_download} bytes/sec (%{size_download} bytes in %{time_total}s)\n  effective URL: %{url_effective}\n" \
    --connect-timeout 5 --max-time 30 \
    https://speed.cloudflare.com/__down?bytes=104857600 2>/dev/null \
    || echo "Download test failed (curl unavailable or network blocked)"

echo "Upload test (10MB to Cloudflare):"
dd if=/dev/zero bs=1M count=10 2>/dev/null | \
    curl -so /dev/null -w "  speed: %{speed_upload} bytes/sec (%{size_upload} bytes in %{time_total}s)\n" \
    --connect-timeout 5 --max-time 30 \
    -T - https://speed.cloudflare.com/__up 2>/dev/null \
    || echo "Upload test failed"

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
