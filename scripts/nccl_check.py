"""Quick NCCL health check — verifies all GPUs can communicate via all_reduce.

Launched by torchrun before the main training/profiling run to catch broken
NCCL setups (SHM corruption, version mismatches, topology issues) early.

Usage (called from shell scripts, not directly):
    torchrun --standalone --nproc_per_node=N scripts/nccl_check.py
"""

import os
import sys

import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
device = torch.device("cuda", rank)

try:
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()

    t = torch.ones(1024, device=device) * (rank + 1)
    dist.all_reduce(t)
    expected = sum(range(1, dist.get_world_size() + 1)) * 1024
    assert abs(t.sum().item() - expected) < 1e-3, (
        f"all_reduce mismatch: {t.sum().item()} != {expected}"
    )

    if rank == 0:
        gpu_name = torch.cuda.get_device_name(0)
        nccl_ver = ".".join(str(x) for x in torch.cuda.nccl.version())
        print(
            f"NCCL OK: {dist.get_world_size()} GPUs communicating "
            f"({gpu_name}, NCCL {nccl_ver})"
        )

    dist.destroy_process_group()

except Exception as e:
    if rank == 0:
        print(f"\nNCCL HEALTH CHECK FAILED: {e}", file=sys.stderr)
        print(file=sys.stderr)
        print("This pod has broken NCCL communication. Common causes:", file=sys.stderr)
        print("  1. NCCL SHM bug -- corrupt /dev/shm/nccl-* segments (seen on some NVL containers)", file=sys.stderr)
        print("  2. NCCL version mismatch with driver/CUDA", file=sys.stderr)
        print("  3. GPU topology not properly exposed to container", file=sys.stderr)
        print(file=sys.stderr)
        print("Quick workaround:  NCCL_SHM_DISABLE=1 bash runs/runpod_profile_comms.sh", file=sys.stderr)
        print("  (WARNING: disabling SHM gives invalid perf numbers on NVL nodes)", file=sys.stderr)
        print(file=sys.stderr)
        print("Recommended: terminate this pod and try a different one.", file=sys.stderr)
    sys.exit(1)
