"""Minimal NVFP4 training for nanochat — block-wise dynamic scaling.

Drop-in replacement for Float8Linear using 4-bit floating point (NVFP4).
Requires Blackwell (SM100+) GPUs — float4_e2m1fn_x2 is not supported on H100.

How NVFP4 differs from FP8
===========================
FP8 uses tensorwise scaling: one FP32 scalar per tensor. This works because FP8
has enough range (e4m3: 8 values per exponent) to represent most distributions.

FP4 has only 1 mantissa bit (e2m1: representable values are {0, 0.5, 1, 1.5, 2,
3, 4, 6}). With just one global scale, outliers would destroy precision for all
other values. So NVFP4 uses two-level "double quantization":

  Level 1 (block scales): One FP8 e4m3 scale per block of 16 elements.
    Each block tracks its own dynamic range independently.

  Level 2 (per-tensor scale): One FP32 scalar normalizes all block scales
    so they use FP8's full range. Without this, block scales for blocks with
    small values would underflow in FP8.

The matmul path is:
  1. Quantize input/weight to FP4 with block-wise FP8 scales
  2. torch._scaled_mm with FP4 data + block scales (cuBLAS Blackwell kernel)
  3. Multiply output by (per_tensor_scale_a * per_tensor_scale_b)

Backward pass uses fake quantization: dequantize to bf16 for gradient matmuls.
This avoids compounding quantization error through the backward pass.

torch._scaled_mm for FP4
=========================
Same function as FP8, but:
  - Data args use torch.float4_e2m1fn_x2 (two FP4 values packed per byte)
  - Scale args are block-wise FP8 tensors (shape: M x K//16), not scalars
  - Scales must be in "swizzled" (blocked) memory layout for cuBLAS
"""

import torch
import torch.nn as nn

from nanochat.common import COMPUTE_DTYPE

BLOCK_SIZE = 16
FP4_E2M1_MAX = 6.0  # max representable value in float4 e2m1
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
EPS = 1e-12


@torch.no_grad()
def _to_nvfp4(x):
    """Quantize a 2D tensor to NVFP4 using block-wise double scaling.

    Args:
        x: Input tensor of shape (M, K) where K must be divisible by BLOCK_SIZE (16).

    Returns:
        (fp4_data, block_scales_fp8, per_tensor_scale) where:
        - fp4_data: shape (M, K) in torch.float4_e2m1fn_x2 (packed, K//2 bytes per row)
        - block_scales_fp8: shape (M, K//16) in torch.float8_e4m3fn
        - per_tensor_scale: scalar FP32 tensor

    TODO(human): Implement the two-level quantization:

    Step 1 — Per-tensor scale (Level 2, computed first):
      tensor_amax = max(|x|) over all elements
      per_tensor_scale = tensor_amax / (FP8_E4M3_MAX * FP4_E2M1_MAX)
      This ensures block scales will fit in FP8 range.

    Step 2 — Block scales (Level 1):
      Reshape x to (M, K//16, 16) blocks
      block_amax = max(|x|) within each block → shape (M, K//16)
      block_scale = block_amax / per_tensor_scale
      Clamp block_scale to [EPS, FP8_E4M3_MAX] and cast to float8_e4m3fn

    Step 3 — Quantize data:
      Compute effective_scale = per_tensor_scale * block_scale_fp8 (broadcast over block)
      x_scaled = x / effective_scale
      Clamp to [-FP4_E2M1_MAX, FP4_E2M1_MAX]
      Cast to torch.float4_e2m1fn_x2

    Hints:
      - Use .unsqueeze(-1) to broadcast block scales over the 16-element blocks
      - per_tensor_scale should stay in float32 (it's applied after the matmul)
      - block_scale_fp8 must be float8_e4m3fn (cuBLAS reads it as FP8)
      - The final cast to float4_e2m1fn_x2 packs two FP4 values per byte
    """
    M, K = x.shape
    assert K % BLOCK_SIZE == 0, f"K={K} must be divisible by BLOCK_SIZE={BLOCK_SIZE}"

    # maximum value over entire tensor
    vmax = x.float().abs().max()
    per_tensor_scale = vmax / (FP8_E4M3_MAX * FP4_E2M1_MAX)
    per_tensor_scale = per_tensor_scale.float()

    # The 16 requirement is due to FP4 Tensorcores reading memory in this format on sm_120 (blackwell) devices
    x = x.reshape(M, K//16, 16)

    # max on dim returns values, indices
    block_vmax = x.float().abs().max(dim=-1).values / per_tensor_scale
    block_vmax = block_vmax.clamp(min=EPS, max=FP8_E4M3_MAX)
    block_scales_fp8 = block_vmax.to(torch.float8_e4m3fn)

    scale = (per_tensor_scale * block_scales_fp8.to(x.dtype)).unsqueeze(-1)
    x_scaled = x / scale
    x_scaled  = x_scaled.clamp(min=-FP4_E2M1_MAX, max=FP4_E2M1_MAX)
    x_fp4 = x_scaled.to(torch.float4_e2m1fn_x2)
    x_fp4 = x_fp4.reshape(M, K)

    return x_fp4, block_scales_fp8, per_tensor_scale.float()


def _to_blocked_scales(scales, M, num_blocks):
    """Rearrange block scales into the swizzled (blocked) layout cuBLAS expects.

    cuBLAS NVFP4 kernels don't read scales in simple row-major order. They
    expect a specific blocked layout that matches how the kernel tiles the
    matmul. This is an implementation detail of NVIDIA's cuBLAS — the exact
    layout may change between CUDA versions.

    Args:
        scales: block scales of shape (M, num_blocks) in float8_e4m3fn
        M: number of rows
        num_blocks: K // BLOCK_SIZE

    Returns:
        Scales rearranged to blocked layout, same number of elements.
    """
    # For now, assume row-major works (it does for some cuBLAS versions).
    # If _scaled_mm errors about scale layout, this needs the swizzle.
    # See torchao.prototype.mx_formats.nvfp4_tensor.to_blocked() for reference.
    return scales.contiguous()


@torch.no_grad()
def _dequantize_nvfp4(fp4_data, block_scales_fp8, per_tensor_scale, orig_dtype):
    """Dequantize NVFP4 back to high precision for backward pass.

    Reverses the quantization: data * block_scale * per_tensor_scale
    """
    M = fp4_data.shape[0]
    K = fp4_data.shape[1]  # logical K (before packing)
    num_blocks = K // BLOCK_SIZE

    # Cast FP4 data back to compute dtype
    data_hp = fp4_data.to(orig_dtype).reshape(M, num_blocks, BLOCK_SIZE)
    # Reconstruct the scale: per_tensor_scale * block_scale
    block_scales = block_scales_fp8.to(orig_dtype)  # (M, num_blocks)
    effective_scale = per_tensor_scale.to(orig_dtype) * block_scales  # (M, num_blocks)
    # Multiply each block by its scale
    data_hp = data_hp * effective_scale.unsqueeze(-1)
    return data_hp.reshape(M, K)


@torch._dynamo.allow_in_graph
class _Float4Matmul(torch.autograd.Function):
    """Custom autograd for NVFP4 forward with fake-quantized backward.

    Forward: quantize input and weight to FP4, matmul via _scaled_mm.
    Backward: dequantize saved tensors to bf16, do gradient matmuls in full precision.
    (This is the standard approach for FP4 — doing backward in FP4 loses too much.)
    """

    @staticmethod
    def forward(ctx, input_2d, weight):
        # Quantize both operands to NVFP4
        in_fp4, in_bscales, in_ptscale = _to_nvfp4(input_2d)
        w_fp4, w_bscales, w_ptscale = _to_nvfp4(weight)

        # Save for backward (we'll dequantize these for gradient computation)
        ctx.save_for_backward(in_fp4, in_bscales, in_ptscale,
                              w_fp4, w_bscales, w_ptscale)
        ctx.orig_dtype = input_2d.dtype

        M, K = input_2d.shape
        N = weight.shape[0]

        # Prepare block scales in the layout cuBLAS expects
        in_scales_blocked = _to_blocked_scales(in_bscales, M, K // BLOCK_SIZE)
        w_scales_blocked = _to_blocked_scales(w_bscales, N, K // BLOCK_SIZE)

        # output = input @ weight.T
        # _scaled_mm with FP4: scales are block-wise FP8, not scalars
        output = torch._scaled_mm(
            in_fp4,
            w_fp4.t(),
            scale_a=in_scales_blocked.view(torch.float8_e4m3fn),
            scale_b=w_scales_blocked.view(torch.float8_e4m3fn),
            out_dtype=input_2d.dtype,
            use_fast_accum=True,
        )

        # Apply per-tensor scales (these are FP32, not baked into _scaled_mm)
        output = output * (in_ptscale * w_ptscale).to(output.dtype)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (in_fp4, in_bscales, in_ptscale,
         w_fp4, w_bscales, w_ptscale) = ctx.saved_tensors

        # Fake quantization backward: dequantize to full precision, then standard matmuls.
        # FP4 has too few representable values to do backward in reduced precision.
        input_hp = _dequantize_nvfp4(in_fp4, in_bscales, in_ptscale, ctx.orig_dtype)
        weight_hp = _dequantize_nvfp4(w_fp4, w_bscales, w_ptscale, ctx.orig_dtype)

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
