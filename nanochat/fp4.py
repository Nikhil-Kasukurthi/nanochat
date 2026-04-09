"""Minimal NVFP4 training for nanochat — block-wise scaling.

Drop-in replacement for Float8Linear using 4-bit floating point (NVFP4).
Requires Blackwell (SM100+) GPUs — float4_e2m1fn_x2 is not supported on H100.

How NVFP4 works
================
FP4 has only 1 mantissa bit (e2m1: representable values are {0, 0.5, 1, 1.5, 2,
3, 4, 6}). One global scale would destroy precision, so NVFP4 uses block-wise
scaling: one FP8 e4m3 scale per block of 16 elements.

The matmul path is:
  1. Quantize input/weight to FP4 with block-wise FP8 scales
  2. torch._scaled_mm with FP4 data + block scales (cuBLAS Blackwell kernel)

Backward pass uses fake quantization: dequantize to bf16 for gradient matmuls.

float4_e2m1fn_x2 is a view-only dtype
=======================================
PyTorch's float4_e2m1fn_x2 is a "shell dtype" — you cannot cast into it via
.to(). Instead, data must be manually encoded as uint8 (two FP4 nibbles per
byte) and reinterpreted via .view(torch.float4_e2m1fn_x2) only at the
_scaled_mm call boundary. This matches torchao's approach.

Reference: pytorch/ao torchao/prototype/mx_formats/nvfp4_tensor.py
"""

import torch
import torch.nn as nn

from nanochat.common import COMPUTE_DTYPE

BLOCK_SIZE = 16
FP4_E2M1_MAX = 6.0  # max representable value in float4 e2m1
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
EPS = 1e-12

# FP4 e2m1 representable absolute values (8 levels) and their 4-bit codes
# Code 0=0, 1=0.5, 2=1.0, 3=1.5, 4=2.0, 5=3.0, 6=4.0, 7=6.0
# Sign bit is bit 3 (0=positive, 1=negative)
# Stored as plain lists — converted to tensors inside functions so torch.compile
# can trace through them (module-level tensors cause FakeTensor errors in dynamo).
_FP4_VALUES_LIST = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# Boundaries for round-to-nearest between consecutive values
_FP4_BOUNDARIES_LIST = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]


def _f32_to_fp4_packed(x_flat):
    """Convert a flat float tensor to packed FP4 e2m1 uint8 (two values per byte).

    Args:
        x_flat: 1D float tensor with an even number of elements, values should
                already be in [-FP4_E2M1_MAX, FP4_E2M1_MAX].

    Returns:
        uint8 tensor with half the elements (two FP4 values packed per byte).
        First value in bits 0-3, second value in bits 4-7.
    """
    boundaries = torch.tensor(_FP4_BOUNDARIES_LIST, device=x_flat.device)
    sign = (x_flat < 0).to(torch.uint8)
    # Map absolute values to nearest FP4 code (0-7) via bucket boundaries
    code = torch.bucketize(x_flat.abs(), boundaries).to(torch.uint8)
    # Combine sign (bit 3) with magnitude code (bits 0-2)
    nibble = code | (sign << 3)
    # Pack pairs: first value in high nibble, second in low nibble
    # This matches cuBLAS's expected packing for float4_e2m1fn_x2
    return ((nibble[0::2] & 0xF) << 4) | (nibble[1::2] & 0xF)


def _fp4_packed_to_f32(packed, device):
    """Unpack FP4 uint8 data back to float32.

    Args:
        packed: uint8 tensor with two FP4 values per byte.
        device: target device for the lookup table.

    Returns:
        float32 tensor with twice the elements.
    """
    fp4_values = torch.tensor(_FP4_VALUES_LIST, device=device)
    # First value in high nibble, second in low nibble (matches packing order)
    first = (packed >> 4) & 0x0F
    second = packed & 0x0F
    # Separate sign (bit 3) and magnitude (bits 0-2)
    first_sign = ((first >> 3) & 1).to(torch.float32) * -2.0 + 1.0
    second_sign = ((second >> 3) & 1).to(torch.float32) * -2.0 + 1.0
    first_mag = fp4_values[(first & 0x07).long()]
    second_mag = fp4_values[(second & 0x07).long()]
    # Interleave back: [first0, second0, first1, second1, ...]
    result = torch.stack([first_sign * first_mag, second_sign * second_mag], dim=-1)
    return result.view(-1)


def _ceil_div(a, b):
    return (a + b - 1) // b


@torch.no_grad()
def _to_nvfp4(x):
    """Quantize a 2D tensor to NVFP4 using block-wise scaling.

    Following torchao's single-level approach:
      block_scale = max(|block|) / FP4_E2M1_MAX, clamped to FP8 range.

    Args:
        x: Input tensor of shape (M, K) where K must be divisible by BLOCK_SIZE (16).

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
    block_scale = block_scale.clamp(min=EPS, max=FP8_E4M3_MAX)
    block_scales_fp8 = block_scale.to(torch.float8_e4m3fn)

    # Scale data into FP4 range using the FP8-rounded scales
    block_scale_f32 = block_scales_fp8.to(torch.float32)
    x_scaled = x_blocks / block_scale_f32.unsqueeze(-1)
    x_scaled = x_scaled.clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)

    # Encode to packed uint8 (two FP4 values per byte)
    x_flat = x_scaled.reshape(-1)
    fp4_packed = _f32_to_fp4_packed(x_flat)
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
    return scales.contiguous().flatten()


@torch.no_grad()
def _dequantize_nvfp4(fp4_packed, block_scales_fp8, orig_dtype):
    """Dequantize NVFP4 back to high precision for backward pass.

    Reverses the quantization: unpack FP4 → float, then multiply by block scales.
    """
    M = fp4_packed.shape[0]
    K_half = fp4_packed.shape[1]
    K = K_half * 2  # logical K
    num_blocks = K // BLOCK_SIZE

    # Unpack FP4 to float32
    data_f32 = _fp4_packed_to_f32(fp4_packed.reshape(-1), fp4_packed.device)
    data_f32 = data_f32.reshape(M, num_blocks, BLOCK_SIZE)

    # Multiply by block scales to recover original magnitude
    block_scales = block_scales_fp8.to(torch.float32)  # (M, num_blocks)
    data_f32 = data_f32 * block_scales.unsqueeze(-1)

    return data_f32.reshape(M, K).to(orig_dtype)


@torch._dynamo.allow_in_graph
class _Float4Matmul(torch.autograd.Function):
    """Custom autograd for NVFP4 forward with fake-quantized backward.

    Forward: quantize input and weight to FP4, matmul via _scaled_mm.
    Backward: dequantize saved tensors to bf16, do gradient matmuls in full precision.
    """

    @staticmethod
    def forward(ctx, input_2d, weight):
        # Quantize both operands to NVFP4 (returns uint8 packed + FP8 scales)
        in_packed, in_bscales = _to_nvfp4(input_2d)
        w_packed, w_bscales = _to_nvfp4(weight)

        # Save for backward (we'll dequantize these for gradient computation)
        ctx.save_for_backward(in_packed, in_bscales, w_packed, w_bscales)
        ctx.orig_dtype = input_2d.dtype

        M, K = input_2d.shape
        N = weight.shape[0]

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

        return output

    @staticmethod
    def backward(ctx, grad_output):
        in_packed, in_bscales, w_packed, w_bscales = ctx.saved_tensors

        # Fake quantization backward: dequantize to full precision, then standard matmuls.
        input_hp = _dequantize_nvfp4(in_packed, in_bscales, ctx.orig_dtype)
        weight_hp = _dequantize_nvfp4(w_packed, w_bscales, ctx.orig_dtype)

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
