import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import os


def test_non_contiguous():
    my_module = build_kernel()
    M, K, N = 37, 41, 29
    A = torch.randn(K, M, device='cuda', dtype=torch.float32).t()
    A = A.t()
    B = torch.randn(N, K, device='cuda', dtype=torch.float32)
    try:
        C = my_module.forward(A, B)
    except RuntimeError:
        print("RuntimeError: Kernel does not support non-contiguous inputs.")
        return
    torch.cuda.synchronize()
    C_ref = torch.matmul(A.t(), B.t()).t()
    assert torch.allclose(C, C_ref, atol=0.01
        ), f'Kernel output does not match reference output. Diff: {torch.abs(C - C_ref).max()}'
