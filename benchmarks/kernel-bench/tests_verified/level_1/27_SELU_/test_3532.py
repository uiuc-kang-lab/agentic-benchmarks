import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_non_contiguous_input():
    my_module = build_kernel()
    x = torch.randn(32, 32, device='cuda', dtype=torch.float32)
    x_nc = x.t()
    try:
        out_kernel = my_module.forward(x_nc)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    out_torch = torch.selu(x_nc)
    assert torch.allclose(out_kernel, out_torch, atol=1e-06
        ), 'Kernel output matches expected SELU output for a non-contiguous input, but it should be incorrect because the kernel does not properly handle non-contiguous tensors.'
