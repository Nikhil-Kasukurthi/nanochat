"""Diagnostic: verify FP4 encoding produces correct IEEE e2m1 codes.

Tests that our inlined IEEE bit-manipulation encoding matches expected values
and produces correct round-trip results through the full quantization pipeline.

Run on B200:
    python tests/test_fp4_encoding.py
"""
import torch


def test_encoding():
    # Test values — all representable FP4 e2m1 magnitudes
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                           -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0])

    from nanochat.fp4 import _f32_to_fp4_packed, _fp4_packed_to_f32
    packed = _f32_to_fp4_packed(values)

    print("Packed bytes:", [f"0x{b:02x}" for b in packed.tolist()])
    print("Packed (binary):", [f"{b:08b}" for b in packed.tolist()])

    # Verify round-trip through nanochat's full pipeline
    from nanochat.fp4 import _to_nvfp4, _dequantize_nvfp4
    M, K = 1, 16
    x = values.reshape(M, K)
    fp4_packed, bscales = _to_nvfp4(x)

    print(f"\nPipeline packed: {[f'0x{b:02x}' for b in fp4_packed.reshape(-1).tolist()]}")

    # Verify dequantization recovers correct values
    x_recovered = _dequantize_nvfp4(fp4_packed, bscales, torch.float32)
    print(f"Recovered: {x_recovered.reshape(-1).tolist()}")

    print(f"\nOriginal: {values.tolist()}")
    print(f"Match: {torch.allclose(values.abs(), x_recovered.reshape(-1).abs(), atol=0.1)}")

    # Optionally compare against torchao if available
    try:
        from torchao.prototype.mx_formats.kernels import f32_to_f4_unpacked, pack_uint4
        torchao_packed = pack_uint4(f32_to_f4_unpacked(values))
        match = torch.equal(packed, torchao_packed)
        print(f"\nBit-exact match with torchao: {match}")
        if not match:
            for i, (o, t) in enumerate(zip(packed.tolist(), torchao_packed.tolist())):
                if o != t:
                    print(f"  byte {i}: ours=0x{o:02x}, torchao=0x{t:02x}")
    except ImportError:
        print("\ntorchao not available for comparison")

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name()}")
        print(f"SM: {torch.cuda.get_device_capability()}")


if __name__ == '__main__':
    test_encoding()
