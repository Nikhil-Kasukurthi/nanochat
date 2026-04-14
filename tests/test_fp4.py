"""Tests for NVFP4 quantization (nanochat/fp4.py).

Tests the quantization math, FP4 encoding, and module conversion logic.
The actual _scaled_mm with float4_e2m1fn_x2 requires Blackwell GPUs,
so we test everything up to that point on any hardware.
"""

import pytest
import torch
import torch.nn as nn

from nanochat.fp4 import (
    _to_nvfp4,
    _dequantize_nvfp4,
    _per_tensor_amax_to_scale,
    _to_blocked_scales,
    Float4Linear,
    convert_to_float4_training,
    FP4_E2M1_MAX,
    FP8_E4M3_MAX,
    FP8_E4M3_TINY,
    BLOCK_SIZE,
)


# --- FP4 encoding/decoding tests ---

def test_fp4_roundtrip_values():
    """FP4 encode -> decode should recover representable values exactly."""
    # All representable FP4 values (positive and negative)
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    # Quantize with unit scale so values pass through unscaled
    M, K = 1, 16
    x = values.reshape(M, K)
    # Use single-level scaling with a known tensor to test encoding
    fp4_packed, _ = _to_nvfp4(x)
    recovered = _dequantize_nvfp4(fp4_packed, torch.ones(M, 1, dtype=torch.float8_e4m3fn), torch.float32)
    assert torch.allclose(values.abs(), recovered.abs(), atol=0.01), f"Expected {values}, got {recovered}"


def test_fp4_rounding():
    """Values between representable FP4 values should round correctly."""
    # 0.3 -> 0.5, 0.6 -> 0.5, 2.4 -> 2.0, 5.5 -> 6.0
    from torchao.prototype.mx_formats.kernels import f32_to_f4_unpacked, f4_unpacked_to_f32
    values = torch.tensor([0.3, 0.6, 2.4, 5.5])
    unpacked = f32_to_f4_unpacked(values)
    recovered = f4_unpacked_to_f32(unpacked)
    expected = torch.tensor([0.5, 0.5, 2.0, 6.0])
    assert torch.allclose(recovered, expected), f"Expected {expected}, got {recovered}"


# --- Quantization math tests ---

def test_block_scales_shape():
    """Block scales should have shape (M, K//16)."""
    M, K = 4, 64
    x = torch.randn(M, K)
    fp4_packed, block_scales = _to_nvfp4(x)
    assert block_scales.shape == (M, K // BLOCK_SIZE)
    assert block_scales.dtype == torch.float8_e4m3fn


def test_packed_data_shape():
    """Packed FP4 data should have shape (M, K//2)."""
    M, K = 4, 64
    x = torch.randn(M, K)
    fp4_packed, _ = _to_nvfp4(x)
    assert fp4_packed.shape == (M, K // 2)
    assert fp4_packed.dtype == torch.uint8


def test_dequantize_roundtrip():
    """Quantize -> dequantize should approximately recover original values."""
    M, K = 8, 64
    x = torch.randn(M, K)
    fp4_packed, block_scales = _to_nvfp4(x)
    x_recovered = _dequantize_nvfp4(fp4_packed, block_scales, torch.float32)

    # FP4 has very few representable values, so relative error will be high
    # but the values should be in the right ballpark (no NaN, similar range)
    assert not x_recovered.isnan().any()
    assert x_recovered.shape == x.shape
    # Mean absolute error should be reasonable (less than the scale of the data)
    mae = (x - x_recovered).abs().mean()
    data_scale = x.abs().mean()
    assert mae < data_scale, f"MAE {mae:.4f} too large vs data scale {data_scale:.4f}"


def test_zero_tensor():
    """All-zeros tensor should not cause division by zero."""
    x = torch.zeros(4, 32)
    fp4_packed, block_scales = _to_nvfp4(x)
    assert not block_scales.to(torch.float32).isnan().any()
    x_recovered = _dequantize_nvfp4(fp4_packed, block_scales, torch.float32)
    assert not x_recovered.isnan().any()


def test_k_not_divisible_by_16():
    """Should raise AssertionError if K is not divisible by BLOCK_SIZE."""
    x = torch.randn(4, 17)
    with pytest.raises(AssertionError):
        _to_nvfp4(x)


# --- Two-level scaling tests ---

def test_two_level_scaling_roundtrip():
    """Two-level scaling quantize -> dequantize should recover values."""
    M, K = 8, 64
    x = torch.randn(M, K) * 10  # larger magnitude to test dynamic range
    pts = _per_tensor_amax_to_scale(x.abs().amax())
    fp4_packed, block_scales = _to_nvfp4(x, per_tensor_scale=pts)
    x_recovered = _dequantize_nvfp4(fp4_packed, block_scales, torch.float32, per_tensor_scale=pts)

    assert not x_recovered.isnan().any()
    assert x_recovered.shape == x.shape
    mae = (x - x_recovered).abs().mean()
    data_scale = x.abs().mean()
    assert mae < data_scale, f"MAE {mae:.4f} too large vs data scale {data_scale:.4f}"


def test_per_tensor_scale_formula():
    """per_tensor_scale = amax / (FP8_E4M3_MAX * FP4_E2M1_MAX)."""
    amax = torch.tensor(100.0)
    pts = _per_tensor_amax_to_scale(amax)
    expected = 100.0 / (448.0 * 6.0)
    assert abs(pts.item() - expected) < 1e-6


# --- Blocked scales tests ---

def test_blocked_scales_element_count():
    """Blocked scales should have M * K//16 elements (flat contiguous)."""
    M, K = 256, 1024
    num_blocks = K // BLOCK_SIZE  # 64
    scales = torch.ones(M, num_blocks, dtype=torch.float8_e4m3fn)
    blocked = _to_blocked_scales(scales, M, num_blocks)

    expected = M * num_blocks  # 256 * 64 = 16384
    assert blocked.numel() == expected, f"Got {blocked.numel()}, expected {expected}"


# --- Module conversion tests ---

def test_convert_to_float4():
    """convert_to_float4_training should replace nn.Linear with Float4Linear."""
    model = nn.Sequential(
        nn.Linear(256, 512, bias=False),
        nn.ReLU(),
        nn.Linear(512, 256, bias=False),
    )

    def filter_fn(mod, fqn):
        return mod.in_features % 16 == 0 and mod.out_features % 16 == 0

    convert_to_float4_training(model, module_filter_fn=filter_fn)

    assert isinstance(model[0], Float4Linear)
    assert isinstance(model[2], Float4Linear)
    assert isinstance(model[1], nn.ReLU)


def test_convert_skips_small_layers():
    """Layers with dims not divisible by 16 should be skipped."""
    model = nn.Sequential(
        nn.Linear(256, 512, bias=False),
        nn.Linear(12, 6, bias=False),
    )

    def filter_fn(mod, fqn):
        return mod.in_features % 16 == 0 and mod.out_features % 16 == 0

    convert_to_float4_training(model, module_filter_fn=filter_fn)

    assert isinstance(model[0], Float4Linear)
    assert not isinstance(model[1], Float4Linear)


def test_convert_shares_weights():
    """Float4Linear should share weight tensors, not copy them."""
    linear = nn.Linear(256, 512, bias=False)
    original_weight = linear.weight
    fp4_linear = Float4Linear.from_float(linear)
    assert fp4_linear.weight is original_weight
