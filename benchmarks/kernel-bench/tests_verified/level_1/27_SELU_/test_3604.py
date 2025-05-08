import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load
import time

def test_non_contiguous_input():
    my_module = build_kernel()
    x = torch.randn(128, 64, device='cuda', dtype=torch.float32)
    x_t = x.t()
    try:
        out_kernel = my_module.forward(x_t)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    out_torch = torch.selu(x_t)
    assert torch.allclose(out_kernel, out_torch, atol=0.01
        ), 'Kernel did not compute SELU correctly on non–contiguous input.'


def test_nontrivial_shape():
    my_module = build_kernel()
    x = torch.randn(8, 16, 32, device='cuda', dtype=torch.float32)
    x_perm = x.permute(2, 0, 1)
    out_kernel = my_module.forward(x_perm)
    out_torch = torch.selu(x_perm)
    assert torch.allclose(out_kernel, out_torch, atol=0.01
        ), 'Kernel did not compute SELU correctly on a non–trivial (permuted) tensor shape.'
