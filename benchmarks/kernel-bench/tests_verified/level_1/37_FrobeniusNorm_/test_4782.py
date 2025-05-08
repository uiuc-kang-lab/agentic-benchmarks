import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_zero_norm_division():
    my_kernel = build_kernel()
    x = torch.zeros(1024, device='cuda', dtype=torch.float32)
    out = my_kernel.forward(x)
    assert torch.isnan(out).any() or torch.isinf(out).any(
        ), 'Expected NaN or Inf in output when input tensor has zero norm'

def test_large_input_multiple_warps():
    my_kernel = build_kernel()
    size = 256 * 70
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    out = my_kernel.forward(x)
    norm = torch.norm(x, p='fro')
    expected = x / norm
    assert torch.allclose(out, expected, atol=0.01
        ), 'Output does not match expected normalization.'
