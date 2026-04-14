"""Minimal NVFP4 training for nanochat — two-level block-wise scaling.

Drop-in replacement for Float8Linear using 4-bit floating point (NVFP4).
Requires Blackwell (SM100+) GPUs — float4_e2m1fn_x2 is not supported on H100.

How NVFP4 works
================
FP4 has only 1 mantissa bit (e2m1: representable values are {0, 0.5, 1, 1.5, 2,
3, 4, 6}). One global scale would destroy precision, so NVFP4 uses two-level
block-wise scaling:
  - Per-tensor scale: amax / (FP8_MAX * FP4_MAX), normalizes dynamic range
  - Per-block scale: one FP8 e4m3 scale per block of 16 elements

The matmul path is:
  1. Quantize input/weight to FP4 with two-level scaling
  2. torch._scaled_mm with FP4 data + block scales (cuBLAS Blackwell kernel)
  3. Multiply output by per-tensor scale product

Backward pass uses fake quantization: dequantize to bf16 for gradient matmuls.
FP4 encoding uses torchao's IEEE round-to-nearest-even bit manipulation.

Reference: pytorch/ao torchao/prototype/mx_formats/nvfp4_tensor.py
"""

import torch
import torch.nn as nn

from torchao.prototype.mx_formats.kernels import (
    f32_to_f4_unpacked, f4_unpacked_to_f32, pack_uint4, unpack_uint4,
)

from nanochat.common import COMPUTE_DTYPE

BLOCK_SIZE = 16
FP4_E2M1_MAX = 6.0  # max representable value in float4 e2m1
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
# Smallest normal FP8 e4m3 value (~1.53e-05). Using 1e-12 is wrong because
# values below tiny flush to zero when cast to FP8, causing inf on reciprocal.
FP8_E4M3_TINY = torch.finfo(torch.float8_e4m3fn).tiny


def _per_tensor_amax_to_scale(amax):
    """Convert per-tensor amax to per-tensor scale for two-level NVFP4 scaling.

    Divides by both FP8_E4M3_MAX and FP4_E2M1_MAX so that block scales can
    utilize the full FP8 e4m3 range when block_max equals tensor_max.
    """
    return amax.float() / (FP8_E4M3_MAX * FP4_E2M1_MAX)


@torch.no_grad()
def _to_nvfp4(x, per_tensor_scale=None):
    """Quantize a 2D tensor to NVFP4 using block-wise scaling.

    Supports two scaling modes:
      - Single-level (per_tensor_scale=None): block_scale = max(|block|) / FP4_E2M1_MAX
      - Two-level (per_tensor_scale given): block scales are normalized by the
        per-tensor scale so they better utilize the FP8 range.

    Args:
        x: Input tensor of shape (M, K) where K must be divisible by BLOCK_SIZE (16).
        per_tensor_scale: Optional global scale from _per_tensor_amax_to_scale().

    Returns:
        (fp4_packed, block_scales_fp8) where:
        - fp4_packed: shape (M, K//2) in uint8 (two FP4 values per byte)
        - block_scales_fp8: shape (M, K//16) in float8_e4m3fn
    """
    M, K = x.shape
    assert K % BLOCK_SIZE == 0, f"K={K} must be divisible by BLOCK_SIZE={BLOCK_SIZE}"

    # Reshape into blocks of 16
    x_blocks = x.float().reshape(M, K // BLOCK_SIZE, BLOCK_SIZE)

    # Block scales: one FP8 scale per block
    block_amax = x_blocks.abs().amax(dim=-1)  # (M, K//16)
    block_scale = block_amax / FP4_E2M1_MAX

    if per_tensor_scale is None:
        # Single-level: quantize block scales directly to FP8
        block_scale = block_scale.clamp(min=FP8_E4M3_TINY, max=FP8_E4M3_MAX)
        block_scales_fp8 = block_scale.to(torch.float8_e4m3fn)
        block_scale_f32 = block_scales_fp8.to(torch.float32)
        x_scaled = x_blocks * (1.0 / block_scale_f32).unsqueeze(-1)
    else:
        # Two-level: normalize block scales by per_tensor_scale before FP8 quantization.
        # This lets block scales use the full FP8 range even when tensor dynamic range is large.
        scaled_block_scales = block_scale / per_tensor_scale
        scaled_block_scales = scaled_block_scales.clamp(min=FP8_E4M3_TINY, max=FP8_E4M3_MAX)
        block_scales_fp8 = scaled_block_scales.to(torch.float8_e4m3fn)
        scaled_block_scales_f32 = block_scales_fp8.to(torch.float32)
        # Combined reciprocal: x * (1/per_tensor_scale) / block_scale, matching MSLK numerics
        reciprocal_scale = (1.0 / per_tensor_scale) / scaled_block_scales_f32
        x_scaled = x_blocks * reciprocal_scale.unsqueeze(-1)

    x_scaled = x_scaled.clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)

    # Encode to packed uint8 using IEEE round-to-nearest-even (two FP4 values per byte)
    fp4_packed = pack_uint4(f32_to_f4_unpacked(x_scaled.reshape(M, K)))
    fp4_packed = fp4_packed.reshape(M, K // 2)

    return fp4_packed, block_scales_fp8


def _to_blocked_scales(scales, M, num_blocks):
    """Rearrange block scales into the swizzled (blocked) layout cuBLAS expects.

    cuBLAS NVFP4 kernels read scales in a specific tiled order. This function
    rearranges (M, num_blocks) scales into that layout.

    The packed FP4 format (float4_e2m1fn_x2) halves the K dimension, so cuBLAS
    sees K_packed = K//2. It expects scales for K_packed/16 = K/32 blocks, but
    we computed K/16 blocks. Each scale must be duplicated to cover both halves
    of a packed block.

    Reference: torchao.prototype.mx_formats.utils.to_blocked()

    Args:
        scales: block scales of shape (M, num_blocks) in float8_e4m3fn
        M: number of rows
        num_blocks: K // BLOCK_SIZE (logical blocks)

    Returns:
        Scales in swizzled blocked layout, flattened to 1D.
    """
    # Use torchao's swizzle which pads to (128, 4) tiles and rearranges
    from torchao.prototype.mx_formats.utils import to_blocked
    return to_blocked(scales)


@torch.no_grad()
def _dequantize_nvfp4(fp4_packed, block_scales_fp8, orig_dtype, per_tensor_scale=None):
    """Dequantize NVFP4 back to high precision for backward pass.

    Reverses the quantization: unpack FP4 → float, then multiply by block scales
    (and per_tensor_scale if two-level scaling was used).
    """
    M = fp4_packed.shape[0]
    K_half = fp4_packed.shape[1]
    K = K_half * 2  # logical K
    num_blocks = K // BLOCK_SIZE

    # Unpack FP4 to float32 using torchao's IEEE-correct conversion
    data_f32 = f4_unpacked_to_f32(unpack_uint4(fp4_packed.reshape(-1)))
    data_f32 = data_f32.reshape(M, num_blocks, BLOCK_SIZE)

    # Multiply by block scales to recover original magnitude
    block_scales = block_scales_fp8.to(torch.float32)  # (M, num_blocks)
    data_f32 = data_f32 * block_scales.unsqueeze(-1)

    # Two-level scaling: also multiply by per_tensor_scale to fully recover magnitude
    if per_tensor_scale is not None:
        data_f32 = data_f32 * per_tensor_scale

    return data_f32.reshape(M, K).to(orig_dtype)


@torch._dynamo.allow_in_graph
class _Float4Matmul(torch.autograd.Function):
    """Custom autograd for NVFP4 forward with fake-quantized backward.

    Forward: quantize input and weight to FP4, matmul via _scaled_mm.
    Backward: dequantize saved tensors to bf16, do gradient matmuls in full precision.
    """

    @staticmethod
    def forward(ctx, input_2d, weight):
        M, K = input_2d.shape
        N = weight.shape[0]

        # Two-level scaling: compute per-tensor scales for better FP8 range utilization
        in_pts = _per_tensor_amax_to_scale(input_2d.abs().amax())
        w_pts = _per_tensor_amax_to_scale(weight.abs().amax())

        # Quantize both operands to NVFP4 with two-level scaling
        in_packed, in_bscales = _to_nvfp4(input_2d, per_tensor_scale=in_pts)
        w_packed, w_bscales = _to_nvfp4(weight, per_tensor_scale=w_pts)

        # Save for backward (we'll dequantize these for gradient computation)
        ctx.save_for_backward(in_packed, in_bscales, w_packed, w_bscales, in_pts, w_pts)
        ctx.orig_dtype = input_2d.dtype

        # Prepare block scales in the layout cuBLAS expects
        in_scales_blocked = _to_blocked_scales(in_bscales, M, K // BLOCK_SIZE)
        w_scales_blocked = _to_blocked_scales(w_bscales, N, K // BLOCK_SIZE)

        # View uint8 packed data as float4_e2m1fn_x2 for _scaled_mm
        # This is a zero-copy reinterpret — never store as float4, only view at call site
        output = torch._scaled_mm(
            in_packed.view(torch.float4_e2m1fn_x2),
            w_packed.view(torch.float4_e2m1fn_x2).t(),
            scale_a=in_scales_blocked.view(torch.float8_e4m3fn),
            scale_b=w_scales_blocked.view(torch.float8_e4m3fn),
            out_dtype=input_2d.dtype,
            use_fast_accum=False,  # not supported for FP4
        )

        # Apply per-tensor scales: output was computed with normalized block scales,
        # so we multiply by the product of per-tensor scales to recover correct magnitude
        output = output * (in_pts * w_pts).to(input_2d.dtype)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        in_packed, in_bscales, w_packed, w_bscales, in_pts, w_pts = ctx.saved_tensors

        # Fake quantization backward: dequantize to full precision, then standard matmuls.
        input_hp = _dequantize_nvfp4(in_packed, in_bscales, ctx.orig_dtype, per_tensor_scale=in_pts)
        weight_hp = _dequantize_nvfp4(w_packed, w_bscales, ctx.orig_dtype, per_tensor_scale=w_pts)

        # grad_input = grad_output @ weight        [B,N] @ [N,K] -> [B,K]
        grad_input = torch.mm(grad_output, weight_hp)
        # grad_weight = grad_output.T @ input      [N,B] @ [B,K] -> [N,K]
        grad_weight = torch.mm(grad_output.t(), input_hp)

        return grad_input, grad_weight


class Float4Linear(nn.Linear):
    """Drop-in nn.Linear replacement that does NVFP4 compute in forward.

    Weights stay in original precision. Only the forward matmul uses FP4.
    Backward uses fake quantization (dequantize -> bf16 matmuls).
    """

    def forward(self, input):
        input = input.to(COMPUTE_DTYPE)
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])
        output = _Float4Matmul.apply(input_2d, self.weight)
        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_float(cls, mod):
        """Create Float4Linear from nn.Linear, sharing weight and bias."""
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


def convert_to_float4_training(module, *, module_filter_fn=None):
    """Replace nn.Linear layers with Float4Linear throughout a module.

    Same tree-walk pattern as convert_to_float8_training. Shares original
    weight/bias tensors — no copies, no extra memory.

    Note: NVFP4 requires K dimension divisible by 16 (BLOCK_SIZE). The
    module_filter_fn should skip layers that don't meet this requirement.
    """
    def _convert(mod, prefix=""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float4Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float4Linear.from_float(child))

    _convert(module)
    return module
