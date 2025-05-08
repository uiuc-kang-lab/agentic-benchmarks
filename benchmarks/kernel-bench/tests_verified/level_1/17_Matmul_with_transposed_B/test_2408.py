import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load

def test_shape_mismatch():
    cuda_module = build_kernel()
    M, K, N = 64, 128, 256
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    with pytest.raises(RuntimeError) as excinfo:
        cuda_module.forward(A, B)
