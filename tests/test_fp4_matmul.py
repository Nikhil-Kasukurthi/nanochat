"""End-to-end FP4 matmul test — requires Blackwell (SM100+) GPU.

Run on B200:
    python tests/test_fp4_matmul.py
"""
import torch

def test_fp4_matmul():
    """Compare FP4 matmul output vs bf16 reference. Should be close, not NaN."""
    from nanochat.fp4 import _to_nvfp4, _to_blocked_scales, BLOCK_SIZE

    torch.manual_seed(42)
    M, K, N = 256, 1024, 512

    input_2d = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)

    # Reference: bf16 matmul
    ref_output = input_2d @ weight.t()

    # FP4 path: quantize (returns uint8 packed + FP8 block scales)
    in_packed, in_bscales = _to_nvfp4(input_2d)
    w_packed, w_bscales = _to_nvfp4(weight)

    print(f"in_packed dtype: {in_packed.dtype}, shape: {in_packed.shape}")
    print(f"in_bscales dtype: {in_bscales.dtype}, shape: {in_bscales.shape}")
    print(f"w_packed dtype: {w_packed.dtype}, shape: {w_packed.shape}")
    print(f"w_bscales dtype: {w_bscales.dtype}, shape: {w_bscales.shape}")

    # Check for NaN in scales
    print(f"in_bscales has NaN: {in_bscales.to(torch.float32).isnan().any()}")
    print(f"w_bscales has NaN: {w_bscales.to(torch.float32).isnan().any()}")

    # Blocked scales
    in_blocked = _to_blocked_scales(in_bscales, M, K // BLOCK_SIZE)
    w_blocked = _to_blocked_scales(w_bscales, N, K // BLOCK_SIZE)
    print(f"in_blocked numel: {in_blocked.numel()}")
    print(f"w_blocked numel: {w_blocked.numel()}")

    # FP4 matmul via _scaled_mm
    output = torch._scaled_mm(
        in_packed.view(torch.float4_e2m1fn_x2),
        w_packed.view(torch.float4_e2m1fn_x2).t(),
        scale_a=in_blocked.view(torch.float8_e4m3fn),
        scale_b=w_blocked.view(torch.float8_e4m3fn),
        out_dtype=torch.bfloat16,
        use_fast_accum=False,
    )

    print(f"\nOutput shape: {output.shape}")
    print(f"Output has NaN: {output.isnan().any()}")
    print(f"Output has Inf: {output.isinf().any()}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"Ref range:    [{ref_output.min():.4f}, {ref_output.max():.4f}]")

    if not output.isnan().any():
        rel_err = (output - ref_output).abs() / (ref_output.abs() + 1e-6)
        print(f"Mean relative error: {rel_err.mean():.4f}")
        print(f"Max relative error:  {rel_err.max():.4f}")
        print("\n✓ FP4 matmul produces valid output")
    else:
        print("\n✗ FP4 matmul produces NaN — scale layout is broken")


def test_fp4_dequant_roundtrip():
    """Check that quantize -> dequantize approximately recovers original values."""
    from nanochat.fp4 import _to_nvfp4, _dequantize_nvfp4

    torch.manual_seed(42)
    M, K = 64, 256
    x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)

    fp4_packed, bscales = _to_nvfp4(x)
    x_recovered = _dequantize_nvfp4(fp4_packed, bscales, torch.bfloat16)

    print(f"\nRoundtrip test (M={M}, K={K}):")
    print(f"Original range:  [{x.min():.4f}, {x.max():.4f}]")
    print(f"Recovered range: [{x_recovered.min():.4f}, {x_recovered.max():.4f}]")
    print(f"Recovered has NaN: {x_recovered.isnan().any()}")

    if not x_recovered.isnan().any():
        rel_err = (x.float() - x_recovered.float()).abs() / (x.float().abs() + 1e-6)
        print(f"Mean relative error: {rel_err.mean():.4f}")
        print(f"Max relative error:  {rel_err.max():.4f}")
        print("✓ Roundtrip produces valid values")
    else:
        print("✗ Roundtrip produces NaN")


if __name__ == '__main__':
    assert torch.cuda.is_available(), "This test requires a CUDA GPU"
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"SM: {torch.cuda.get_device_capability()}\n")

    test_fp4_dequant_roundtrip()
    print("\n" + "="*60 + "\n")
    test_fp4_matmul()
