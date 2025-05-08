import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load

def test_invalid_dimensions():
    my_module = build_kernel()
    batch_size = 2
    M, K1, K2, N = 4, 8, 7, 3
    A = torch.randn(batch_size, M, K1, dtype=torch.float32, device='cuda')
    B = torch.randn(batch_size, K2, N, dtype=torch.float32, device='cuda')
    with pytest.raises(RuntimeError):
        my_module.forward(A, B)
