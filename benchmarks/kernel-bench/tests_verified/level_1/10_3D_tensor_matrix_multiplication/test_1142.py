import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_shared_memory_comment_is_misleading():
    my_module = build_kernel()
    N, M, K, L = 4, 8, 16, 10
    A = torch.randn(N, M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, L, device='cuda', dtype=torch.float32)
    output = my_module.forward(A, B)
    ref = torch.matmul(A, B)
    assert torch.allclose(output, ref, atol=0.01
        ), f'Kernel output does not match torch.matmul output. Max diff: {(output - ref).abs().max().item()}'


def test_input_dimension_broadcasting_failure():
    my_module = build_kernel()
    M, K, L = 8, 16, 10
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, L, device='cuda', dtype=torch.float32)
    try:
        result = my_module.forward(A, B)
    except Exception:
        print("Exception: Kernel does not support non-broadcasting types.")
        return
    torch.cuda.synchronize()
    ref = torch.matmul(A, B)
    assert torch.allclose(result, ref, atol=0.01
        ), f'Kernel output does not match reference output. Max diff: {(result - ref).abs().max().item()}'


def test_input_dimension_extra_dims_failure():
    my_module = build_kernel()
    N, M, K, L = 2, 4, 8, 10
    A = torch.randn(1, N, M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, L, device='cuda', dtype=torch.float32)
    with pytest.raises(Exception):
        my_module.forward(A, B)
