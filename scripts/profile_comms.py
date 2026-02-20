"""
Profile communication overhead in nanochat's distributed optimizer.

All inter-GPU communication in nanochat is concentrated in DistMuonAdamW.step()
via a 3-phase async pipeline. This script instruments those phases with CUDA
events and NVTX markers to measure exactly how much time is spent communicating.

Usage:
    torchrun --nproc_per_node=8 -m scripts.profile_comms -- --depth 12
    nsys profile -o comms torchrun --nproc_per_node=8 -m scripts.profile_comms -- --depth 12
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import types
import argparse
from datetime import datetime, timezone

import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.tokenizer import get_tokenizer

# -----------------------------------------------------------------------------
# CLI

parser = argparse.ArgumentParser(description="Profile comm overhead in distributed optimizer")
parser.add_argument("--depth", type=int, default=12)
parser.add_argument("--aspect-ratio", type=int, default=64)
parser.add_argument("--head-dim", type=int, default=128)
parser.add_argument("--num-steps", type=int, default=20, help="profiled steps")
parser.add_argument("--warmup-steps", type=int, default=10)
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--output-dir", type=str, default="profile_output")
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (matches speedrun)")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Init

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

if not (ddp and device_type == "cuda"):
    print0("Comm profiling requires multi-GPU CUDA. Use: torchrun --nproc_per_node=N")
    compute_cleanup()
    raise SystemExit(0)

autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# -----------------------------------------------------------------------------
# Model + optimizer (same setup as base_train.py)

tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

base_dim = args.depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
config = GPTConfig(
    sequence_len=args.max_seq_len, vocab_size=vocab_size,
    n_layer=args.depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
)
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

num_params = sum(p.numel() for p in model.parameters())
print0(f"Model: d{args.depth} ({num_params:,} params) | dim={model_dim} | heads={num_heads}")

# FP8 conversion (must happen before torch.compile)
if args.fp8:
    from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
    import torch.nn as nn
    def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        if min(mod.in_features, mod.out_features) < 128:
            return False
        return True
    fp8_config = Float8LinearConfig.from_recipe_name("tensorwise")
    num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
    num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
    print0(f"FP8 enabled: converted {num_fp8}/{num_linear} linear layers")

model = torch.compile(model, dynamic=False)
optimizer = model.setup_optimizer()

# Batch size
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len * ddp_world_size
total_batch_size = 2**19  # 524288, standard for d12
if total_batch_size % tokens_per_fwdbwd != 0:
    total_batch_size = tokens_per_fwdbwd
grad_accum_steps = total_batch_size // tokens_per_fwdbwd
print0(f"Batch: {total_batch_size:,} tokens | grad_accum={grad_accum_steps}")

# Dataloader
train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device,
)
x, y = next(train_loader)

# -----------------------------------------------------------------------------
# Monkey-patch DistMuonAdamW.step() with phase instrumentation
#
# The 3 phases in DistMuonAdamW (see optim.py):
#   Phase 1: Fire all async reduce_scatter / all_reduce
#   Phase 2: Wait for reduces, compute update on owned shard, fire all_gather
#   Phase 3: Wait for all gathers, copy params back

phase_timings = []  # populated by instrumented_step
_current_step = [0]  # mutable container so instrumented_step can read it

@torch.no_grad()
def instrumented_step(self):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_profiled = _current_step[0] >= args.warmup_steps

    evt_start = torch.cuda.Event(enable_timing=True)
    evt_p1 = torch.cuda.Event(enable_timing=True)
    evt_p2 = torch.cuda.Event(enable_timing=True)
    evt_p3 = torch.cuda.Event(enable_timing=True)
    evt_start.record()

    # Phase 1: launch all async reduce ops
    if is_profiled:
        torch.cuda.nvtx.range_push("Phase1-Reduces")
    reduce_infos = []
    for group in self.param_groups:
        if group['kind'] == 'adamw':
            reduce_infos.append(self._reduce_adamw(group, world_size))
        elif group['kind'] == 'muon':
            reduce_infos.append(self._reduce_muon(group, world_size))
    if is_profiled:
        torch.cuda.nvtx.range_pop()
    evt_p1.record()

    # Phase 2: wait, compute, launch gathers
    if is_profiled:
        torch.cuda.nvtx.range_push("Phase2-Compute+Gather")
    gather_list = []
    for group, info in zip(self.param_groups, reduce_infos):
        if group['kind'] == 'adamw':
            self._compute_adamw(group, info, gather_list, rank, world_size)
        elif group['kind'] == 'muon':
            self._compute_muon(group, info, gather_list, rank)
    if is_profiled:
        torch.cuda.nvtx.range_pop()
    evt_p2.record()

    # Phase 3: wait for gathers, copy back
    if is_profiled:
        torch.cuda.nvtx.range_push("Phase3-WaitGathers")
    self._finish_gathers(gather_list)
    if is_profiled:
        torch.cuda.nvtx.range_pop()
    evt_p3.record()

    torch.cuda.synchronize()
    phase_timings.append({
        "phase1_ms": evt_start.elapsed_time(evt_p1),
        "phase2_ms": evt_p1.elapsed_time(evt_p2),
        "phase3_ms": evt_p2.elapsed_time(evt_p3),
        "total_ms": evt_start.elapsed_time(evt_p3),
    })

optimizer.step = types.MethodType(instrumented_step, optimizer)

# -----------------------------------------------------------------------------
# Profiling loop

total_steps = args.warmup_steps + args.num_steps
step_times = []

# PyTorch Profiler for Chrome traces — skip under nsys (CUPTI conflict)
under_nsys = "NSYS_PROFILING_SESSION_ID" in os.environ
if not under_nsys:
    if master_process:
        os.makedirs(args.output_dir, exist_ok=True)
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=args.warmup_steps, active=args.num_steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.output_dir) if master_process else None,
        record_shapes=True,
    )
    profiler.start()
else:
    print0("Running under nsys — PyTorch Profiler disabled (CUPTI conflict)")
    profiler = None

print0(f"\nRunning {total_steps} steps ({args.warmup_steps} warmup + {args.num_steps} profiled)...")

for step in range(total_steps):
    _current_step[0] = step  # tell instrumented_step whether to emit NVTX markers
    is_warmup = step < args.warmup_steps

    # Signal nsys to start/stop capture at the warmup boundary.
    # Requires nsys --capture-range=cudaProfilerApi to take effect.
    # Without that flag, nsys captures everything (these calls are harmless no-ops).
    if step == args.warmup_steps:
        torch.cuda.cudart().cudaProfilerStart()

    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        (loss / grad_accum_steps).backward()
        x, y = next(train_loader)

    optimizer.step()
    model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    dt_ms = (time.time() - t0) * 1000

    if not is_warmup:
        step_times.append(dt_ms)
    if profiler is not None:
        profiler.step()

    label = "warmup" if is_warmup else "prof"
    print0(f"  step {step:3d} [{label:6s}] {dt_ms:.1f} ms")

torch.cuda.cudart().cudaProfilerStop()

if profiler is not None:
    profiler.stop()

# -----------------------------------------------------------------------------
# Results

profiled_phases = phase_timings[args.warmup_steps:]
avg_step = sum(step_times) / len(step_times)
avg_p1 = sum(t["phase1_ms"] for t in profiled_phases) / len(profiled_phases)
avg_p2 = sum(t["phase2_ms"] for t in profiled_phases) / len(profiled_phases)
avg_p3 = sum(t["phase3_ms"] for t in profiled_phases) / len(profiled_phases)
avg_opt = sum(t["total_ms"] for t in profiled_phases) / len(profiled_phases)

# Console summary
print0(f"\n{'='*50}")
print0(f"  GPU: {torch.cuda.get_device_name(0)} x {ddp_world_size}")
print0(f"  Model: d{args.depth} ({num_params:,} params)")
print0(f"  Avg step:       {avg_step:7.1f} ms")
print0(f"  Optimizer step: {avg_opt:7.1f} ms ({avg_opt/avg_step*100:.1f}%)")
print0(f"    Phase 1 (reduces):        {avg_p1:7.1f} ms ({avg_p1/avg_step*100:.1f}%)")
print0(f"    Phase 2 (compute+gather): {avg_p2:7.1f} ms ({avg_p2/avg_step*100:.1f}%)")
print0(f"    Phase 3 (wait gathers):   {avg_p3:7.1f} ms ({avg_p3/avg_step*100:.1f}%)")
print0(f"{'='*50}")

# JSON dump
if master_process:
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "metadata": {
            "gpu_name": torch.cuda.get_device_name(0),
            "num_gpus": ddp_world_size,
            "model_depth": args.depth,
            "num_params": num_params,
            "device_batch_size": args.device_batch_size,
            "max_seq_len": args.max_seq_len,
            "total_batch_size": total_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "num_steps_profiled": args.num_steps,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "measured": {
            "avg_step_ms": round(avg_step, 2),
            "avg_optimizer_step_ms": round(avg_opt, 2),
            "avg_phase1_ms": round(avg_p1, 2),
            "avg_phase2_ms": round(avg_p2, 2),
            "avg_phase3_ms": round(avg_p3, 2),
            "comm_overhead_pct": round(avg_opt / avg_step * 100, 2),
            "per_step": [
                {
                    "step": i,
                    "step_ms": round(step_times[i], 2),
                    "phase1_ms": round(profiled_phases[i]["phase1_ms"], 2),
                    "phase2_ms": round(profiled_phases[i]["phase2_ms"], 2),
                    "phase3_ms": round(profiled_phases[i]["phase3_ms"], 2),
                }
                for i in range(len(profiled_phases))
            ],
        },
    }
    json_path = os.path.join(args.output_dir, "profile_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print0(f"Wrote {json_path}")

compute_cleanup()
