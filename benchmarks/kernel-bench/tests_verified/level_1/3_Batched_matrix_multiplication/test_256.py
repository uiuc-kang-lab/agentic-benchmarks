import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load



def test_non_aligned_memory():
    module = build_kernel()
    batch_size, m, k, n = 2, 33, 33, 33
    A_big = torch.randn(batch_size, m, k + 1, device='cuda', dtype=torch.
        float32)
    B_big = torch.randn(batch_size, k + 1, n, device='cuda', dtype=torch.
        float32)
    A = A_big.narrow(2, 1, k).contiguous()
    B = B_big.narrow(1, 1, k).contiguous()
    C = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.bmm(A, B)
    assert torch.allclose(C, C_ref, atol=0.01
        ), f'Kernel output does not match reference output. Diff: {torch.abs(C - C_ref).max()}'
