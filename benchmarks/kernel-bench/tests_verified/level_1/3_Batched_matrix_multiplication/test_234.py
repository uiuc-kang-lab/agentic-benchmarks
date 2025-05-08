import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
batch_size = 16
M = 16
K = 32
N = 16


def test_unaligned_input():
    cuda_mod = build_kernel()
    A_storage = torch.randn(batch_size * M * K + 1, device='cuda', dtype=
        torch.float32)
    A_unaligned = A_storage.narrow(0, 1, batch_size * M * K).view(batch_size,
        M, K)
    B = torch.randn(batch_size, K, N, device='cuda', dtype=torch.float32)
    C_kernel = cuda_mod.forward(A_unaligned, B)
    torch.cuda.synchronize()
    C_ref = torch.bmm(A_unaligned.contiguous(), B.contiguous())
    assert torch.allclose(C_kernel, C_ref, atol=0.01
        ), f'Result of kernel does not match reference. Diff: {torch.abs(C_kernel - C_ref).max()}'


def test_kernel_launch_error_checking():
    cuda_mod = build_kernel()
    A = torch.randn(batch_size, M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(batch_size, K, N, device='cuda', dtype=torch.float32)
    C = cuda_mod.forward(A, B)
    torch.cuda.synchronize()
    assert C.shape == (batch_size, M, N
        ), 'Output tensor C has an unexpected shape.'
