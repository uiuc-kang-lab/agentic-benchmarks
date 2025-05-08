import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_dimension_mismatch():
    cuda_module = build_kernel()
    N, M, K, L = 4, 8, 16, 10
    A = torch.randn(N, M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K + 1, L, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        C_kernel = cuda_module.forward(A, B)


if __name__ == '__main__':
    pytest.main([__file__])
