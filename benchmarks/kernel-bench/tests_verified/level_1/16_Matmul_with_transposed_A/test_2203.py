import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def test_noncontiguous_inputs():
    my_module = build_kernel()
    M = 128
    K = 256
    N = 128
    A_base = torch.randn(K, M * 2, device='cuda', dtype=torch.float32)
    B_base = torch.randn(K, N * 2, device='cuda', dtype=torch.float32)
    A = A_base[:, ::2]
    B = B_base[:, ::2]
    C_ref = torch.matmul(A.t(), B)
    try:
        C = my_module.forward(A, B)
    except RuntimeError:
        print("RuntimeError: Kernel does not support non-contiguous inputs.")
        return
    torch.cuda.synchronize()
    assert torch.allclose(C, C_ref, atol=0.01
        ), 'Kernel did not fail on non-contiguous inputs as expected.'


def test_kernel_dimension_mismatch():
    my_module = build_kernel()
    M = 64
    K = 128
    N = 32
    A = torch.randn(K, M, device='cuda', dtype=torch.float32)
    B = torch.randn(K + 1, N, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()


def test_tiled_kernel_correctness():
    my_module = build_kernel()
    M = 1024
    K = 512
    N = 1024
    A = torch.randn(K, M, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    C_ref = torch.matmul(A.t(), B)
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    assert torch.allclose(C, C_ref, atol=0.01
        ), f'Tiled kernel output mismatch. Max error: {(C - C_ref).abs().max().item()}'


if __name__ == '__main__':
    pytest.main([__file__])
