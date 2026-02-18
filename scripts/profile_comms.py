"""
Profile communication overhead in nanochat's distributed optimizer.

nanochat does NOT use PyTorch DDP — all inter-GPU communication is manually
orchestrated inside DistMuonAdamW.step() via a 3-phase async pipeline.
Zero communication happens during forward/backward; it's all concentrated
in the optimizer step. This makes profiling clean and targeted.

Usage:
    # Standard run (PyTorch Profiler traces + summary; NVTX markers present but dormant)
    torchrun --nproc_per_node=8 -m scripts.profile_comms -- --depth 12

    # Also capture Nsight traces — just prepend nsys, same script
    nsys profile -o comms_profile torchrun --nproc_per_node=8 -m scripts.profile_comms -- --depth 12

    # Quick local test (single GPU / CPU / MPS — no comm profiling, zero overhead)
    python -m scripts.profile_comms --depth 4 --num-steps 3 --device-batch-size 2 --max-seq-len 512
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import math
import time
import types
import argparse
from datetime import datetime, timezone
from contextlib import nullcontext

import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import (
    compute_init, compute_cleanup, print0, autodetect_device_type,
)
from nanochat.tokenizer import get_tokenizer

# -----------------------------------------------------------------------------
# CLI arguments

parser = argparse.ArgumentParser(description="Profile communication overhead in distributed optimizer")
parser.add_argument("--depth", type=int, default=12, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--num-steps", type=int, default=10, help="number of profiling steps")
parser.add_argument("--warmup-steps", type=int, default=3, help="warmup steps (excluded from measurements)")
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--output-dir", type=str, default="profile_output", help="directory for output files")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Compute init

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

# Device check: should we profile comms?
profile_comms = ddp and device_type == "cuda"
if not profile_comms:
    print0("Single device or non-CUDA — running training loop only (no comm profiling overhead)")
    print0("Use torchrun for multi-GPU profiling.")

autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

# -----------------------------------------------------------------------------
# Model init

tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

base_dim = args.depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim

config = GPTConfig(
    sequence_len=args.max_seq_len,
    vocab_size=vocab_size,
    n_layer=args.depth,
    n_head=num_heads,
    n_kv_head=num_heads,
    n_embd=model_dim,
)

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

num_params = sum(p.numel() for p in model.parameters())
print0(f"Model: d{args.depth} ({num_params:,} params) | dim={model_dim} | heads={num_heads}")

model = torch.compile(model, dynamic=False)

# -----------------------------------------------------------------------------
# Optimizer

optimizer = model.setup_optimizer()

# Batch size and gradient accumulation
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
# Use a fixed total_batch_size = 2^19 = 524288 (standard for d12 profiling)
total_batch_size = 2**19
# Ensure divisibility; if not possible, adjust
if total_batch_size % world_tokens_per_fwdbwd != 0:
    # Fall back to 1 grad accum step
    total_batch_size = world_tokens_per_fwdbwd
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Batch: {total_batch_size:,} tokens | grad_accum_steps={grad_accum_steps} | device_batch_size={args.device_batch_size}")

# -----------------------------------------------------------------------------
# Dataloader

train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device
)
x, y = next(train_loader)

# -----------------------------------------------------------------------------
# Section 2: Analytical communication volume calculator

def compute_comm_volume(optimizer, world_size):
    """
    Compute exact bytes transferred per optimizer step from param group shapes.
    Returns dict with per-group and total volumes.
    """
    groups = []
    total_bytes = 0

    for group in optimizer.param_groups:
        params = group["params"]
        if not params:
            continue

        kind = group["kind"]
        if kind == "adamw":
            group_bytes = 0
            for p in params:
                elem_size = p.element_size()
                if p.numel() < 1024:
                    # Small params: all_reduce only
                    vol = p.numel() * elem_size
                    group_bytes += vol
                else:
                    # Large params: reduce_scatter + all_gather
                    vol = 2 * p.numel() * elem_size
                    group_bytes += vol

            shape_str = "mixed" if len(set(p.shape for p in params)) > 1 else str(params[0].shape)
            groups.append({
                "name": f"adamw_{shape_str}",
                "kind": "adamw",
                "num_params": len(params),
                "bytes": group_bytes,
            })
            total_bytes += group_bytes

        elif kind == "muon":
            # Muon: K same-shape params stacked, padded to ceil(K/world_size)*world_size
            K = len(params)
            shape = params[0].shape
            elem_size = params[0].element_size()
            chunk_size = math.ceil(K / world_size)
            padded_K = chunk_size * world_size
            # reduce_scatter + all_gather on the stacked tensor
            vol = 2 * padded_K * shape[0] * shape[1] * elem_size
            groups.append({
                "name": f"muon_{shape[0]}x{shape[1]}",
                "kind": "muon",
                "num_params": K,
                "padded_num_params": padded_K,
                "bytes": vol,
            })
            total_bytes += vol

    # Bus bandwidth correction: ring algorithm transfers (N-1)/N of total
    bus_correction = (world_size - 1) / world_size if world_size > 1 else 1.0
    bus_volume = int(total_bytes * bus_correction)

    return {
        "per_group": groups,
        "comm_volume_bytes": total_bytes,
        "bus_volume_bytes": bus_volume,
    }


def compute_theoretical_times(bus_volume_bytes):
    """Compute expected comm time for various interconnects."""
    interconnects = {
        "pcie_gen4_x16": 32e9,    # 32 GB/s
        "pcie_gen5_x16": 64e9,    # 64 GB/s
        "nvlink_h100_sxm": 900e9, # 900 GB/s
        "nvl72": 900e9,           # 900 GB/s per GPU
    }
    return {
        name: (bus_volume_bytes / bw) * 1000  # ms
        for name, bw in interconnects.items()
    }


# Compute analytical volumes (even for single GPU — useful info)
comm_info = compute_comm_volume(optimizer, ddp_world_size)
theoretical_times = compute_theoretical_times(comm_info["bus_volume_bytes"])

if master_process:
    print0(f"\n{'='*60}")
    print0(f"Analytical Communication Volume")
    print0(f"{'='*60}")
    for g in comm_info["per_group"]:
        vol_gb = g["bytes"] / 1e9
        print0(f"  {g['name']:30s} | {g['kind']:6s} | {g['num_params']:3d} params | {vol_gb:.3f} GB")
    print0(f"  {'':30s} | {'Total':6s} | {' ':10s} | {comm_info['comm_volume_bytes']/1e9:.3f} GB")
    print0(f"  {'':30s} | {'Bus':6s}   | {' ':10s} | {comm_info['bus_volume_bytes']/1e9:.3f} GB")
    print0(f"\nTheoretical comm time per step:")
    for name, t_ms in theoretical_times.items():
        print0(f"  {name:25s} | {t_ms:.2f} ms")
    print0(f"{'='*60}\n")

# -----------------------------------------------------------------------------
# Section 3: Monkey-patch DistMuonAdamW.step() with phase instrumentation

phase_timings = []  # list of dicts, one per step

if profile_comms:
    from nanochat.optim import DistMuonAdamW

    original_step = optimizer.step.__func__  # unbound method

    def instrumented_step(self):
        """Instrumented version of DistMuonAdamW.step() with 3-phase timing."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # CUDA events for precise GPU-side timing
        evt_start = torch.cuda.Event(enable_timing=True)
        evt_p1_end = torch.cuda.Event(enable_timing=True)
        evt_p2_end = torch.cuda.Event(enable_timing=True)
        evt_p3_end = torch.cuda.Event(enable_timing=True)

        evt_start.record()

        # --- Phase 1: Launch all async reduce ops ---
        with torch.profiler.record_function("Phase1-ReduceScatter"):
            torch.cuda.nvtx.range_push("Phase1-ReduceScatter")
            reduce_infos = []
            for group in self.param_groups:
                if group['kind'] == 'adamw':
                    reduce_infos.append(self._reduce_adamw(group, world_size))
                elif group['kind'] == 'muon':
                    reduce_infos.append(self._reduce_muon(group, world_size))
            torch.cuda.nvtx.range_pop()

        evt_p1_end.record()

        # --- Phase 2: Wait for reduces, compute updates, launch gathers ---
        with torch.profiler.record_function("Phase2-Compute+Gather"):
            torch.cuda.nvtx.range_push("Phase2-Compute+Gather")
            gather_list = []
            for group, info in zip(self.param_groups, reduce_infos):
                if group['kind'] == 'adamw':
                    self._compute_adamw(group, info, gather_list, rank, world_size)
                elif group['kind'] == 'muon':
                    self._compute_muon(group, info, gather_list, rank)
            torch.cuda.nvtx.range_pop()

        evt_p2_end.record()

        # --- Phase 3: Wait for gathers, copy back ---
        with torch.profiler.record_function("Phase3-WaitGathers"):
            torch.cuda.nvtx.range_push("Phase3-WaitGathers")
            self._finish_gathers(gather_list)
            torch.cuda.nvtx.range_pop()

        evt_p3_end.record()

        # Synchronize and read timings
        torch.cuda.synchronize()
        phase_timings.append({
            "phase1_ms": evt_start.elapsed_time(evt_p1_end),
            "phase2_ms": evt_p1_end.elapsed_time(evt_p2_end),
            "phase3_ms": evt_p2_end.elapsed_time(evt_p3_end),
            "total_ms": evt_start.elapsed_time(evt_p3_end),
        })

    # Apply monkey-patch
    optimizer.step = types.MethodType(instrumented_step, optimizer)
    print0("Monkey-patched DistMuonAdamW.step() with phase instrumentation")

# -----------------------------------------------------------------------------
# Section 4: Profiling loop

num_steps = args.num_steps
warmup_steps = args.warmup_steps
total_steps = warmup_steps + num_steps
step_times = []  # wall-clock per step (post-warmup only)

print0(f"\nRunning {total_steps} steps ({warmup_steps} warmup + {num_steps} profiled)...")

if profile_comms:
    # Wrap with torch.profiler for Chrome/TensorBoard traces
    profiler_schedule = torch.profiler.schedule(
        wait=0,
        warmup=warmup_steps,
        active=num_steps,
        repeat=1,
    )
    output_dir = args.output_dir
    if master_process:
        os.makedirs(output_dir, exist_ok=True)

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=profiler_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir) if master_process else None,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    )
    profiler.start()
else:
    profiler = None

for step in range(total_steps):
    is_warmup = step < warmup_steps

    synchronize()
    t0 = time.time()

    # Forward + backward (grad accumulation)
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        loss = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)

    # Optimizer step (all communication happens here)
    with torch.profiler.record_function("OptimizerStep") if profile_comms else nullcontext():
        optimizer.step()
    model.zero_grad(set_to_none=True)

    synchronize()
    t1 = time.time()
    dt_ms = (t1 - t0) * 1000

    if not is_warmup:
        step_times.append(dt_ms)

    if profiler is not None:
        profiler.step()

    status = "warmup" if is_warmup else "profiled"
    print0(f"  step {step:3d} [{status:8s}] | {dt_ms:.1f} ms")

if profiler is not None:
    profiler.stop()

print0(f"\nProfiling complete.")

# -----------------------------------------------------------------------------
# Section 5-6: Bandwidth utilization report and structured output

if not step_times:
    print0("No profiled steps — nothing to report.")
    compute_cleanup()
    raise SystemExit(0)

avg_step_ms = sum(step_times) / len(step_times)

# Phase timings (only available in multi-GPU CUDA mode)
measured = {
    "avg_step_ms": round(avg_step_ms, 2),
    "per_step": [],
}

if profile_comms and phase_timings:
    # Discard warmup phase timings
    profiled_phases = phase_timings[warmup_steps:]

    if profiled_phases:
        avg_p1 = sum(t["phase1_ms"] for t in profiled_phases) / len(profiled_phases)
        avg_p2 = sum(t["phase2_ms"] for t in profiled_phases) / len(profiled_phases)
        avg_p3 = sum(t["phase3_ms"] for t in profiled_phases) / len(profiled_phases)
        avg_phase_total = sum(t["total_ms"] for t in profiled_phases) / len(profiled_phases)
        comm_pct = (avg_phase_total / avg_step_ms * 100) if avg_step_ms > 0 else 0

        measured.update({
            "avg_phase1_ms": round(avg_p1, 2),
            "avg_phase2_ms": round(avg_p2, 2),
            "avg_phase3_ms": round(avg_p3, 2),
            "avg_optimizer_step_ms": round(avg_phase_total, 2),
            "comm_overhead_pct": round(comm_pct, 2),
        })

        for i, t in enumerate(profiled_phases):
            measured["per_step"].append({
                "step": i,
                "step_ms": round(step_times[i], 2),
                "phase1_ms": round(t["phase1_ms"], 2),
                "phase2_ms": round(t["phase2_ms"], 2),
                "phase3_ms": round(t["phase3_ms"], 2),
            })

        # Parse profiler key averages for NCCL kernel time
        if profiler is not None:
            try:
                key_avgs = profiler.key_averages()
                nccl_time_us = sum(
                    evt.cuda_time_total for evt in key_avgs
                    if "nccl" in evt.key.lower()
                )
                compute_time_us = sum(
                    evt.cuda_time_total for evt in key_avgs
                    if "nccl" not in evt.key.lower() and evt.cuda_time_total > 0
                )
                measured["nccl_kernel_time_ms"] = round(nccl_time_us / 1000 / num_steps, 2)
                measured["compute_kernel_time_ms"] = round(compute_time_us / 1000 / num_steps, 2)
            except Exception as e:
                print0(f"Warning: Could not parse profiler key averages: {e}")
else:
    # Single device: just record per-step wall-clock
    for i, dt in enumerate(step_times):
        measured["per_step"].append({"step": i, "step_ms": round(dt, 2)})

# Build structured output
gpu_name = torch.cuda.get_device_name(0) if device_type == "cuda" else device_type
results = {
    "metadata": {
        "gpu_name": gpu_name,
        "num_gpus": ddp_world_size,
        "model_depth": args.depth,
        "num_params": num_params,
        "device_batch_size": args.device_batch_size,
        "max_seq_len": args.max_seq_len,
        "grad_accum_steps": grad_accum_steps,
        "total_batch_size": total_batch_size,
        "num_steps_profiled": num_steps,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    },
    "analytical": {
        "comm_volume_bytes": comm_info["comm_volume_bytes"],
        "per_group": comm_info["per_group"],
        "bus_volume_bytes": comm_info["bus_volume_bytes"],
        "theoretical_times_ms": {k: round(v, 2) for k, v in theoretical_times.items()},
    },
    "measured": measured,
}

# Write output files (rank 0 only)
if master_process:
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # JSON output
    json_path = os.path.join(output_dir, "profile_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print0(f"Wrote {json_path}")

    # Markdown report
    md_lines = []
    md_lines.append(f"# Communication Profiling: {gpu_name} ({ddp_world_size} GPUs)")
    md_lines.append(f"**Model:** d{args.depth} ({num_params:,} params) | **Batch:** {total_batch_size:,} tokens | **Date:** {datetime.now().strftime('%Y-%m-%d')}")
    md_lines.append("")

    # Analytical estimates
    md_lines.append("## Analytical Estimates")
    md_lines.append("| Collective | Volume | Notes |")
    md_lines.append("|---|---|---|")
    for g in comm_info["per_group"]:
        vol_str = f"{g['bytes']/1e9:.2f} GB" if g['bytes'] > 1e6 else f"{g['bytes']/1e3:.2f} KB"
        note = f"{g['num_params']} params"
        if g["kind"] == "muon":
            note += f" (padded to {g.get('padded_num_params', g['num_params'])})"
        md_lines.append(f"| {g['name']} ({g['kind']}) | {vol_str} | {note} |")
    md_lines.append(f"| **Total per step** | **{comm_info['comm_volume_bytes']/1e9:.2f} GB** | bus volume: {comm_info['bus_volume_bytes']/1e9:.2f} GB |")
    md_lines.append("")

    # Measured timing
    md_lines.append(f"## Measured Timing (avg over {len(step_times)} steps)")
    md_lines.append("| Phase | Time (ms) | % of Step |")
    md_lines.append("|---|---|---|")
    md_lines.append(f"| Full step | {avg_step_ms:.1f} | 100% |")

    if profile_comms and "avg_phase1_ms" in measured:
        for phase_name, key in [("Phase 1 (reduces)", "avg_phase1_ms"),
                                 ("Phase 2 (compute + gather)", "avg_phase2_ms"),
                                 ("Phase 3 (wait gathers)", "avg_phase3_ms")]:
            val = measured[key]
            pct = val / avg_step_ms * 100 if avg_step_ms > 0 else 0
            md_lines.append(f"| {phase_name} | {val:.1f} | {pct:.1f}% |")
        opt_ms = measured["avg_optimizer_step_ms"]
        opt_pct = measured["comm_overhead_pct"]
        md_lines.append(f"| **Optimizer step total** | **{opt_ms:.1f}** | **{opt_pct:.1f}%** |")
        if "nccl_kernel_time_ms" in measured:
            nccl_pct = measured["nccl_kernel_time_ms"] / avg_step_ms * 100 if avg_step_ms > 0 else 0
            md_lines.append(f"| NCCL kernel time (profiler) | {measured['nccl_kernel_time_ms']:.1f} | {nccl_pct:.1f}% |")
    md_lines.append("")

    # Projected training overhead
    md_lines.append("## Projected 3-Hour Training Overhead")
    md_lines.append("| Interconnect | Comm Time / Step | Overhead | Est. Added Time |")
    md_lines.append("|---|---|---|---|")
    training_hours = 3.0
    training_seconds = training_hours * 3600
    interconnect_display = {
        "pcie_gen4_x16": "PCIe Gen4",
        "pcie_gen5_x16": "PCIe Gen5",
        "nvlink_h100_sxm": "H100 SXM (NVLink)",
        "nvl72": "NVL72",
    }
    for key, display in interconnect_display.items():
        t_comm = theoretical_times[key]
        # Estimate overhead: replace measured comm time with theoretical
        if avg_step_ms > 0:
            overhead_pct = t_comm / avg_step_ms * 100
            # Added time over a 3-hour run
            added_min = (t_comm / 1000) * (training_seconds / (avg_step_ms / 1000)) / 60
            md_lines.append(f"| {display} | {t_comm:.1f} ms | {overhead_pct:.1f}% | +{added_min:.1f} min |")
    md_lines.append("")

    md_path = os.path.join(output_dir, "profile_results.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print0(f"Wrote {md_path}")

    # Print the summary to console
    print0(f"\n{'='*60}")
    print0("\n".join(md_lines))
    print0(f"{'='*60}")

# Cleanup
compute_cleanup()
