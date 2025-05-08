import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load

def test_dimension_mismatch_issue():
    mod = build_kernel()
    M, K, N = 128, 32, 128
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K + 1, N, dtype=torch.float32, device='cuda')
    with pytest.raises(Exception):
        _ = torch.matmul(A, B)
