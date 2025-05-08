import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import os


def test_unused_shared_memory():
    batch_size = 1
    num_channels = 1
    input_length = 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = False
    x = torch.randn(batch_size, num_channels, input_length, device='cuda',
        dtype=torch.float32)
    expected_length = (input_length + 2 * padding - dilation * (kernel_size -
        1) - 1) // stride + 1
    custom_module = build_kernel()
    y = custom_module.forward(x, kernel_size, stride, padding, dilation,
        return_indices)
    assert y.shape == (batch_size, num_channels, expected_length
        ), 'Output shape does not match expected shape despite unused shared memory.'

def test_return_indices_concatenation():
    batch_size = 1
    num_channels = 1
    input_length = 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = True
    x = torch.randn(batch_size, num_channels, input_length, device='cuda',
        dtype=torch.float32)
    custom_module = build_kernel()
    out = custom_module.forward(x, kernel_size, stride, padding, dilation,
        return_indices)
    expected_length = (input_length + 2 * padding - dilation * (kernel_size -
        1) - 1) // stride + 1
    expected_shape = batch_size, num_channels, expected_length * 2
    assert out.shape == expected_shape, 'Return indices concatenation issue: output shape does not match expected concatenated shape.'

if __name__ == '__main__':
    pytest.main([__file__])
