"""Per-rank NUMA binding wrapper for torchrun-spawned processes.

torchrun always launches children as `python <script> <args>`. This script
reads LOCAL_RANK from the environment, computes the correct NUMA node, and
re-execs the real training command under `numactl --cpunodebind --membind`.

Usage (called by torchrun, not directly):
    torchrun --nproc_per_node=8 scripts/numa_wrapper.py python -m scripts.profile_comms -- --depth 12

Requires:
    GPUS_PER_NODE  env var (e.g. 4 for 8 GPUs / 2 sockets)
    LOCAL_RANK     set automatically by torchrun
"""

import os
import sys

local_rank = int(os.environ["LOCAL_RANK"])
gpus_per_node = int(os.environ.get("GPUS_PER_NODE", "4"))
numa_node = local_rank // gpus_per_node

# Replace this process with numactl wrapping the real command.
# sys.argv[1:] is whatever follows numa_wrapper.py, e.g. ["python", "-m", "scripts.profile_comms", ...]
args = ["numactl", f"--cpunodebind={numa_node}", f"--membind={numa_node}"] + sys.argv[1:]
os.execvp("numactl", args)
