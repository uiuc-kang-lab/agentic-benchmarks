import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load
import numpy as np


def test_intra_block_reduction():
    my_module = build_kernel()
    M = 64
    K = 5000
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, 1, device='cuda', dtype=torch.float32)
    C1 = my_module.forward(A, B)
    torch.cuda.synchronize()
    C2 = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    assert torch.allclose(C1, C_ref, atol=0.01
        ), f'Kernel output C1 deviates from reference! Maximum diff: {(C1 - C_ref).abs().max()}'
    assert torch.allclose(C2, C_ref, atol=0.01
        ), f'Kernel output C2 deviates from reference! Maximum diff: {(C2 - C_ref).abs().max()}'

def test_launch_configuration_general_input():
    my_module = build_kernel()
    M = 103
    K = 128
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, 1, device='cuda', dtype=torch.float32)
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    assert torch.allclose(C, C_ref, atol=0.01
        ), f'Kernel output differs from reference output! Max diff: {(C - C_ref).abs().max()}'


if __name__ == '__main__':
    pytest.main([__file__])
