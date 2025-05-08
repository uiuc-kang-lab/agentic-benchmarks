import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load


def reference_forward(A, B):
    return torch.diag(A) @ B


def test_vectorized_grid_configuration():
    N = 32
    M = 512
    A = torch.randn(N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, M, device='cuda', dtype=torch.float32)
    ref = reference_forward(A, B)
    module = build_kernel()
    out = module.forward(A, B)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=0.01
        ), 'Test must detect the grid configuration issue in vectorized branch.'


def test_multiple_kernel_launches():
    N = 64
    M = 300
    A = torch.randn(N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, M, device='cuda', dtype=torch.float32)
    ref = reference_forward(A, B)
    module = build_kernel()
    out = module.forward(A, B)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=0.01
        ), 'Test must detect interference from multiple kernel launches.'


def test_thread_count_calculation():
    N = 16
    M = 10
    A = torch.randn(N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, M, device='cuda', dtype=torch.float32)
    ref = reference_forward(A, B)
    module = build_kernel()
    out = module.forward(A, B)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=0.01
        ), 'Test must detect issues caused by the thread count miscalculation.'
