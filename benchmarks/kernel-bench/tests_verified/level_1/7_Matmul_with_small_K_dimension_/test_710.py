import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load

def test_dimension_mismatch():
    my_module = build_kernel()
    M, K, N = 128, 16, 128
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K + 1, N, dtype=torch.float32, device='cuda')
    with pytest.raises(Exception):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

if __name__ == '__main__':
    pytest.main([__file__])
