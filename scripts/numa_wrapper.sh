#!/bin/bash
# numa_wrapper.sh — Per-rank NUMA binding for torchrun-spawned processes.
#
# torchrun sets LOCAL_RANK in each child process. This script maps
# LOCAL_RANK to the correct NUMA node and execs the real command
# under numactl --cpunodebind + --membind.
#
# Usage (called by torchrun, not directly):
#   torchrun --nproc_per_node=8 scripts/numa_wrapper.sh python -m scripts.profile_comms -- --depth 12
#
# Requires:
#   GPUS_PER_NODE  — GPUs per NUMA node (e.g. 4 for 8 GPUs / 2 sockets)
#   LOCAL_RANK     — set automatically by torchrun

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NUMA_NODE=$((LOCAL_RANK / GPUS_PER_NODE))
exec numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE "$@"
