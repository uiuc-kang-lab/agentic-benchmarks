import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import os


def test_A_cpu_transfer_behavior():
    N = 1024
    M = 512
    A = torch.randn(N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, M, device='cuda', dtype=torch.float32)
    module = build_kernel()
    C = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.diag(A) @ B
    assert torch.allclose(C, C_ref, atol=0.01
        ), 'Kernel unexpectedly handled GPU tensor A without transferring!'

if __name__ == '__main__':
    pytest.main()
