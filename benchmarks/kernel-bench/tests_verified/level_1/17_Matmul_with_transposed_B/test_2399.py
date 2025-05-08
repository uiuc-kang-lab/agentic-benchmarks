import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def test_incorrect_B_layout():
    my_module = build_kernel()
    M, K, N = 64, 128, 32
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(N, K, dtype=torch.float32, device='cuda')
    B_incorrect = B.T
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B_incorrect)
        torch.cuda.synchronize()

if __name__ == '__main__':
    pytest.main([__file__])
