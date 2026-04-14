"""NVFP4 training for nanochat via NVIDIA TransformerEngine.

Uses TransformerEngine's te.Linear with NVFP4BlockScaling recipe for production-grade
FP4 training on Blackwell GPUs. This replaces our custom FP4 implementation with
NVIDIA's fused CUDA kernels that handle:
  - Two-level block scaling (16-element blocks + global FP32 scale)
  - 2D weight quantization (16×16 blocks for weights)
  - Stochastic rounding for gradients
  - Random Hadamard Transform for wgrad
  - FP4 GEMMs in all three passes (forward, grad_input, grad_weight)
  - Full torch.compile compatibility (proper custom ops with FakeTensor support)

Requirements:
  pip install --no-build-isolation transformer_engine[pytorch]
  Requires TransformerEngine >= 2.7 for NVFP4BlockScaling support.

Reference: https://developer.nvidia.com/blog/using-nvfp4-low-precision-model-training-for-higher-throughput-without-losing-accuracy
"""

import torch.nn as nn

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling


def get_nvfp4_recipe():
    """Return the NVFP4BlockScaling recipe for use with te.autocast."""
    return NVFP4BlockScaling()


def convert_to_float4_training(module, *, module_filter_fn=None):
    """Replace nn.Linear layers with te.Linear throughout a module.

    te.Linear is a drop-in replacement that uses NVFP4 when wrapped in
    te.autocast(recipe=NVFP4BlockScaling()). Outside autocast, it behaves
    like a normal nn.Linear in the params_dtype.

    Args:
        module: Root module to convert.
        module_filter_fn: Optional filter(module, fqn) -> bool. Only matching
            Linears are converted. Common use: skip layers with dims not
            divisible by 16.
    """
    def _convert(mod, prefix=""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, te.Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    # Create te.Linear sharing the same weight
                    te_linear = te.Linear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        params_dtype=child.weight.dtype,
                        device="meta",  # avoid VRAM spike
                    )
                    # Share weight and bias tensors (no copies)
                    te_linear.weight = child.weight
                    if child.bias is not None:
                        te_linear.bias = child.bias
                    setattr(mod, name, te_linear)

    _convert(module)
    return module
