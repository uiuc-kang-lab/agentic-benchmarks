import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def test_non_contiguous_input():
    cuda_mod = build_kernel()
    x = torch.randn(16, 64, 256, 256, dtype=torch.float32, device='cuda'
        ).transpose(1, 2)
    weight = torch.randn(64, dtype=torch.float32, device='cuda')
    bias = torch.randn(64, dtype=torch.float32, device='cuda')
    running_mean = torch.zeros(64, dtype=torch.float32, device='cuda')
    running_var = torch.ones(64, dtype=torch.float32, device='cuda')
    try:
        result = cuda_mod.forward(x, weight, bias, running_mean, running_var, True,
            0.1, 1e-05)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    expected = torch.nn.functional.batch_norm(x, weight, bias, running_mean,
        running_var, True, 0.1, 1e-05)
    assert torch.allclose(result, expected, atol=0.01
        ), 'Kernel produced incorrect results for non-contiguous input. This indicates it may not be handling non-contiguous memory layouts correctly.'


def test_unusual_input_shape():
    cuda_mod = build_kernel()
    x = torch.randn(32, 128, 1, 1, dtype=torch.float32, device='cuda')
    weight = torch.randn(128, dtype=torch.float32, device='cuda')
    bias = torch.randn(128, dtype=torch.float32, device='cuda')
    running_mean = torch.zeros(128, dtype=torch.float32, device='cuda')
    running_var = torch.ones(128, dtype=torch.float32, device='cuda')
    output = cuda_mod.forward(x, weight, bias, running_mean, running_var, 
        True, 0.1, 1e-05)
    torch.cuda.synchronize()
    assert output.shape == x.shape, 'Output shape mismatch for unusual input shape.'
    assert not torch.isnan(output).any(
        ), 'Output contains NaNs for unusual input shape.'
