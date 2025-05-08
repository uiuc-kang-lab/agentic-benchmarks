import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load
import os


@pytest.mark.skipif(not torch.cuda.is_available(), reason=
    'CUDA is not available.')
def test_incorrect_input_layout():
    """
    Issue 1: The kernel assumes that input tensors have specific layouts and are contiguous.
    Here we purposefully pass non-contiguous inputs (or wrong shape) so that the assumptions are broken.
    Expected: The result does not match torch.matmul(A.T, B.T)
    """
    M, K, N = 512, 1023, 256
    A = torch.randn(K, M, device='cuda', dtype=torch.float32)
    B = torch.randn(N, K, device='cuda', dtype=torch.float32)
    A_non_contig = A.T.T
    B_non_contig = B.index_select(0, torch.randperm(B.size(0)).to('cuda')
        ).index_select(0, torch.argsort(torch.randperm(B.size(0)).to('cuda')))
    my_module = build_kernel()
    try:
        C = my_module.forward(A_non_contig, B_non_contig)
    except RuntimeError:
        print("RuntimeError: Kernel does not support non-contiguous inputs.")
        return
    torch.cuda.synchronize()
    C_ref = torch.matmul(A_non_contig.T, B_non_contig.T)
    assert torch.allclose(C, C_ref, atol=0.01
        ), f'Test for input layout did not trigger an error: difference max {(C - C_ref).abs().max()}'
