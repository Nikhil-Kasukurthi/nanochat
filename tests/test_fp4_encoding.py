"""Diagnostic: verify FP4 encoding produces correct IEEE e2m1 codes.

Now that nanochat uses torchao's f32_to_f4_unpacked directly, this test
validates that the full quantization pipeline (scale + encode + pack)
produces correct results.

Run on B200:
    python tests/test_fp4_encoding.py
"""
import torch


def test_encoding():
    # Test values — all representable FP4 e2m1 magnitudes
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                           -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0])

    from torchao.prototype.mx_formats.kernels import f32_to_f4_unpacked, pack_uint4
    unpacked = f32_to_f4_unpacked(values)
    packed = pack_uint4(unpacked)

    print("Packed bytes:", [f"0x{b:02x}" for b in packed.tolist()])
    print("Packed (binary):", [f"{b:08b}" for b in packed.tolist()])

    # Verify round-trip through nanochat's full pipeline
    from nanochat.fp4 import _to_nvfp4, _dequantize_nvfp4
    M, K = 1, 16
    x = values.reshape(M, K)
    fp4_packed, bscales = _to_nvfp4(x)

    # With unit-scale block, the encoded values should match torchao's direct encoding
    print(f"\nPipeline packed: {[f'0x{b:02x}' for b in fp4_packed.reshape(-1).tolist()]}")

    # Verify dequantization recovers correct values
    x_recovered = _dequantize_nvfp4(fp4_packed, bscales, torch.float32)
    print(f"Recovered: {x_recovered.reshape(-1).tolist()}")

    # Check that the values are correct (within block scale precision)
    print(f"\nOriginal: {values.tolist()}")
    print(f"Match: {torch.allclose(values.abs(), x_recovered.reshape(-1).abs(), atol=0.1)}")

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name()}")
        print(f"SM: {torch.cuda.get_device_capability()}")


if __name__ == '__main__':
    test_encoding()
