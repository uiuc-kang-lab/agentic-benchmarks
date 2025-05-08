import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import os


@pytest.fixture(scope='module')
def kernel_module():
    return build_kernel()


def test_return_indices_behavior(kernel_module):
    batch_size = 2
    channels = 3
    seq_len = 16
    x = torch.randn(batch_size, channels, seq_len, device='cuda', dtype=
        torch.float32)
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = True
    out = kernel_module.forward(x, kernel_size, stride, padding, dilation,
        return_indices)
    expected_output_length = (seq_len + 2 * padding - dilation * (
        kernel_size - 1) - 1) // stride + 1
    assert out.size(-1
        ) != expected_output_length, f'Expected concatenated output when return_indices is True. Got last dimension size {out.size(-1)} equal to expected pooling output length {expected_output_length}.'
    torch.cuda.synchronize()


def test_kernel_size_runtime(kernel_module):
    batch_size = 2
    channels = 3
    seq_len = 32
    x = torch.randn(batch_size, channels, seq_len, device='cuda', dtype=
        torch.float32)
    kernel_size = 7
    stride = 2
    padding = 3
    dilation = 1
    return_indices = False
    out = kernel_module.forward(x, kernel_size, stride, padding, dilation,
        return_indices)
    expected_output_length = (seq_len + 2 * padding - dilation * (
        kernel_size - 1) - 1) // stride + 1
    assert out.shape == (batch_size, channels, expected_output_length
        ), f'Unexpected output shape: got {out.shape}, expected ({batch_size}, {channels}, {expected_output_length}).'
    torch.cuda.synchronize()


def test_grid_calculation(kernel_module):
    batch_size = 128
    channels = 64
    seq_len = 512
    x = torch.randn(batch_size, channels, seq_len, device='cuda', dtype=
        torch.float32)
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = False
    out = kernel_module.forward(x, kernel_size, stride, padding, dilation,
        return_indices)
    expected_output_length = (seq_len + 2 * padding - dilation * (
        kernel_size - 1) - 1) // stride + 1
    assert out.shape == (batch_size, channels, expected_output_length
        ), f'Unexpected output shape: got {out.shape}, expected ({batch_size}, {channels}, {expected_output_length}).'
    torch.cuda.synchronize()
