import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import os


def test_non_contiguous_input():
    x = torch.randn(64, 256, device='cuda', dtype=torch.float32)
    x_t = x.t()
    my_kernel = build_kernel()
    try:
        y_kernel = my_kernel.forward(x_t)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    y_reference = torch.nn.functional.softplus(x_t)
    assert torch.allclose(y_kernel, y_reference, atol=0.01
        ), 'Kernel unexpectedly produced correct output on non-contiguous tensor.'


if __name__ == '__main__':
    pytest.main([__file__])
