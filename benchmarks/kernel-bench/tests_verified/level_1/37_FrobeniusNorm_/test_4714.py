import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import os
import math
import pytest
import torch
from torch.utils.cpp_extension import load

def test_non_contiguous_tensor():
    my_module = build_kernel()
    x = torch.randn(16, 64, 256, 256, dtype=torch.float32, device='cuda')
    x_noncontig = x.transpose(1, 2)
    try:
        result = my_module.forward(x_noncontig)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    expected = x_noncontig / torch.norm(x_noncontig, p='fro')
    assert torch.allclose(result, expected, atol=0.01
        ), 'Kernel produced incorrect results for non-contiguous input. This indicates it may not be handling non-contiguous memory layouts correctly.'


def test_zero_norm():
    my_module = build_kernel()
    x = torch.zeros(16, 64, 256, 256, dtype=torch.float32, device='cuda')
    output = my_module.forward(x)
    assert torch.isnan(output).any(
        ), 'Output should contain NaNs when input norm is zero'


def test_shared_memory_issue():
    my_module = build_kernel()
    x = torch.randn(16, 64, 16, 16, dtype=torch.float32, device='cuda')
    x_shared = x.share_memory_()
    result = my_module.forward(x_shared)
    expected = x_shared / torch.norm(x_shared, p='fro')
    assert torch.allclose(result, expected, atol=0.01
        ), 'Kernel produced incorrect results for shared memory input. This indicates it may not be handling shared memory correctly.'
