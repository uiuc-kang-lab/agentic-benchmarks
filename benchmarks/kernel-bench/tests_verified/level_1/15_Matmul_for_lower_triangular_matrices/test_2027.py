import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_large_matrix_tile_index_inversion():
    N = 8192
    A = torch.randn(N, N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, N, device='cuda', dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    mod = build_kernel()
    C = mod.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.tril(torch.matmul(A, B))
    assert torch.allclose(C, C_ref, atol=0.01
        ), f'Large matrix multiplication mismatch: max diff {(C - C_ref).abs().max().item()}'
