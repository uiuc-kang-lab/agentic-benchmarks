import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_non_contiguous_input():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    N, M = 16, 17
    x = torch.randn(N, M, dtype=torch.float32, device='cuda')
    x = x.t()
    my_module = build_kernel()
    try:
        y = my_module.forward(x)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    x_ref = torch.tanh(x)
    torch.cuda.synchronize()
    assert torch.allclose(y, x_ref, atol=0.01
        ), 'Kernel did not expose misalignment issue on non-contiguous input.'
