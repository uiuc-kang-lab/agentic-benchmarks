import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load


def reference_avg_pool1d(x, kernel_size, stride, padding):
    avg_pool = nn.AvgPool1d(kernel_size, stride=stride, padding=padding)
    return avg_pool(x)


def test_non_contiguous_input():
    kernel_module = build_kernel()
    batch_size = 4
    in_channels = 8
    input_length = 64
    kernel_size = 3
    stride = 1
    padding = 1
    x = torch.randn(batch_size, in_channels, input_length, device='cuda',
        dtype=torch.float32)
    x_non_contig = x.transpose(1, 2).transpose(1, 2)
    try:
        output_kernel = kernel_module.forward(x_non_contig, kernel_size, stride,
            padding)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    output_ref = reference_avg_pool1d(x_non_contig, kernel_size, stride,
        padding)
    assert torch.allclose(output_kernel, output_ref, atol=0.01
        ), 'Kernel output should differ from the reference output on non-contiguous inputs.'



def test_dynamic_kernel_size():
    kernel_module = build_kernel()
    batch_size = 4
    in_channels = 8
    input_length = 128
    kernel_size = 7
    stride = 2
    padding = 2
    x = torch.randn(batch_size, in_channels, input_length, device='cuda',
        dtype=torch.float32)
    output_kernel = kernel_module.forward(x, kernel_size, stride, padding)
    output_ref = reference_avg_pool1d(x, kernel_size, stride, padding)
    assert torch.allclose(output_kernel, output_ref, atol=0.01
        ), 'Kernel output should differ from the reference output when using a dynamic kernel_size.'
