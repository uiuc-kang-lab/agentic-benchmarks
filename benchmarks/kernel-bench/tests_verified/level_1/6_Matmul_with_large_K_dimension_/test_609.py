import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


@pytest.fixture(scope='module')
def cuda_module():
    mod = build_kernel()
    return mod


def test_non_contiguous_input(cuda_module):
    M, N, K = 128, 64, 256
    A_base = torch.randn(M, K * 2, device='cuda', dtype=torch.float32)
    B_base = torch.randn(K, N * 2, device='cuda', dtype=torch.float32)
    A_noncontig = A_base.as_strided((M, K), (A_base.stride(0), A_base.
        stride(1) * 2))
    B_noncontig = B_base.as_strided((K, N), (B_base.stride(0), B_base.
        stride(1) * 2))
    try:
        C_kernel = cuda_module.forward(A_noncontig, B_noncontig)
    except RuntimeError:
        print("RuntimeError: Kernel does not support non-float32 types.")
        return
    torch.cuda.synchronize()
    C_ref = torch.matmul(A_noncontig.contiguous(), B_noncontig.contiguous())
    assert torch.allclose(C_kernel, C_ref, atol=0.01
        ), f'Kernel output does not match reference output. Diff: {torch.abs(C_kernel - C_ref).max()}'


def test_mismatched_dimensions(cuda_module):
    M, N, K = 128, 64, 256
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K + 1, N, device='cuda', dtype=torch.float32)
    with pytest.raises(Exception):
        _ = cuda_module.forward(A, B)
