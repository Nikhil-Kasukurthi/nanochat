"""Low-precision training for nanochat via NVIDIA TransformerEngine.

Supports NVFP4 and MXFP8 block-scaled recipes on Blackwell GPUs.
TransformerEngine provides fused CUDA kernels with full torch.compile support.

Recipes:
  NVFP4 — 4-bit matmuls with 16-element blocks, 2D weight scaling, RHT, stochastic rounding
  MXFP8 — 8-bit matmuls with 32-element blocks, E8M0 block scales

Requirements:
  pip install --no-build-isolation transformer_engine[pytorch]
  TransformerEngine >= 2.7 for block scaling recipes.
"""

import torch
import torch.nn as nn

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling, MXFP8BlockScaling


def get_te_recipe(name="nvfp4"):
    """Return a TransformerEngine recipe for use with te.autocast.

    Args:
        name: 'nvfp4' for FP4 block scaling, 'mxfp8' for MXFP8 block scaling.
    """
    if name == "nvfp4":
        return NVFP4BlockScaling()
    elif name == "mxfp8":
        return MXFP8BlockScaling()
    else:
        raise ValueError(f"Unknown recipe: {name}. Use 'nvfp4' or 'mxfp8'.")


# Keep backward compat
def get_nvfp4_recipe():
    return get_te_recipe("nvfp4")


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
                    # Create te.Linear with bf16 params (te handles mixed precision internally).
                    # nanochat's Linear keeps fp32 master weights, but te.Linear expects
                    # params_dtype to match input dtype. Copy weight to bf16 — the optimizer
                    # will maintain fp32 master weights in its state dict.
                    te_linear = te.Linear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        params_dtype=torch.bfloat16,
                        device=child.weight.device if not child.weight.is_meta else "meta",
                    )
                    te_linear.weight.data = child.weight.data.to(torch.bfloat16)
                    if child.bias is not None:
                        te_linear.bias.data = child.bias.data.to(torch.bfloat16)
                    setattr(mod, name, te_linear)

    _convert(module)
    return module
