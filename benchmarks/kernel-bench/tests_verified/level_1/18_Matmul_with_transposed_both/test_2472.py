import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import os


def test_dimension_mismatch():
    M, K, N = 32, 64, 32
    A = torch.randn(K, M, device='cuda', dtype=torch.float32)
    B = torch.randn(N, K + 1, device='cuda', dtype=torch.float32)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = module.forward(A, B)
        torch.cuda.synchronize()

if __name__ == '__main__':
    pytest.main([os.path.realpath(__file__)])
