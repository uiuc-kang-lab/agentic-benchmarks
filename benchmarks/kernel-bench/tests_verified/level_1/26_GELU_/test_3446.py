import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


def test_multiple_kernel_launches():
    mod = build_kernel()
    N = 16384 + 16
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y = mod.forward(x)
    y_ref = F.gelu(x)
    assert torch.allclose(y, y_ref, atol=0.01
        ), 'Output incorrect when multiple kernel launches occur.'


def test_remainder_handling():
    mod = build_kernel()
    N = 1023
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y = mod.forward(x)
    y_ref = F.gelu(x)
    assert torch.allclose(y, y_ref, atol=0.01
        ), 'Output incorrect for tensors with remainder elements.'


if __name__ == '__main__':
    pytest.main([__file__])
