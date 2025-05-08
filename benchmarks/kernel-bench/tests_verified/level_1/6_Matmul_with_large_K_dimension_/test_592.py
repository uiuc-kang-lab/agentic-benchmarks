import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load
import os


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_misaligned_memory():
    M, K, N = 256, 512, 256
    A_full = torch.randn(M + 1, K, device='cuda', dtype=torch.float32)
    B_full = torch.randn(K, N, device='cuda', dtype=torch.float32)
    A = A_full[1:]
    ext = build_kernel()
    C = ext.forward(A, B_full)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B_full)
    assert torch.allclose(C, C_ref, atol=0.01
        ), f'Kernel output does not match reference output. Diff: {torch.abs(C - C_ref).max()}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_dimension_mismatch_error():
    M, K, N = 256, 512, 256
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K + 1, N, device='cuda', dtype=torch.float32)
    ext = build_kernel()
    with pytest.raises(Exception):
        C = ext.forward(A, B)
        torch.cuda.synchronize()

if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
