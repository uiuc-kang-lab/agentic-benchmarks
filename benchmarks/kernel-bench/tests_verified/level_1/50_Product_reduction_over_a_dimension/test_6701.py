import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load
kernel_module = build_kernel()


def run_forward(x, dim):
    return kernel_module.forward(x, dim)

def test_contiguous_check():
    x = torch.randn(16, 32, 32, device='cuda', dtype=torch.float32).transpose(
        0, 1)
    try:
        result = run_forward(x, 1)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    result_ref = torch.prod(x, dim=1)
    assert torch.allclose(result, result_ref, atol=0.01
        ), f'Expected {result_ref}, got {result}'
