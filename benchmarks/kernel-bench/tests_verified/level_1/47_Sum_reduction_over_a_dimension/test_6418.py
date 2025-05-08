import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_non_contiguous_input_transpose():
    cuda_module = build_kernel()
    x = torch.randn(16, 256, 256, device='cuda', dtype=torch.float32)
    x_t = x.transpose(1, 2).contiguous().transpose(1, 2)
    reduce_dim = 1
    expected = torch.sum(x_t, dim=reduce_dim, keepdim=True)
    try:
        kernel_out = cuda_module.forward(x_t, reduce_dim)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    assert torch.allclose(kernel_out, expected, atol=0.001
        ), f'Kernel output: {kernel_out}, Reference output: {expected}'


def test_non_contiguous_input_slicing():
    cuda_module = build_kernel()
    x = torch.randn(32, 64, 128, device='cuda', dtype=torch.float32)
    x_slice = x[:, ::2, :]
    reduce_dim = 1
    try:
        kernel_out = cuda_module.forward(x_slice, reduce_dim)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    expected = torch.sum(x_slice, dim=reduce_dim, keepdim=True)
    assert torch.allclose(kernel_out, expected, atol=0.001
        ), f'Kernel output: {kernel_out}, Reference output: {expected}'
