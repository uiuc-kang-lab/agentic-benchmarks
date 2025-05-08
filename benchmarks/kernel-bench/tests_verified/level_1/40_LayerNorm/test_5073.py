import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_non_flat_normalized_shape():
    my_module = build_kernel()
    batch = 2
    norm_shape = 4, 8
    normalized_size = 4 * 8
    x = torch.randn(batch, *norm_shape, device='cuda', dtype=torch.float32)
    weight = torch.randn(*norm_shape, device='cuda', dtype=torch.float32)
    bias = torch.randn(*norm_shape, device='cuda', dtype=torch.float32)
    ref = torch.nn.LayerNorm(norm_shape, elementwise_affine=True).to(
        device='cuda', dtype=torch.float32)(x)
    out = my_module.forward(x, weight, bias)
    assert torch.allclose(out, ref, atol=0.01
        ), 'Kernel produced incorrect results for non-flat normalized shape. This indicates it may not be handling non-contiguous memory layouts correctly.'
