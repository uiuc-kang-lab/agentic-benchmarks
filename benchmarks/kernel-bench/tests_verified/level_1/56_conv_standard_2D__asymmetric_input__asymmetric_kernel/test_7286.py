import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import torch.nn.functional as F
import pytest
from torch.utils.cpp_extension import load
import math


def ref_conv2d(input, weight, bias, stride, padding, dilation, groups):
    return F.conv2d(input, weight, bias=bias, stride=stride, padding=
        padding, dilation=dilation, groups=groups)


@pytest.fixture(scope='module')
def cuda_kernel_module():
    module = build_kernel()
    return module


def test_shared_memory_usage(cuda_kernel_module):
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = 5, 7
    stride = 1, 1
    padding = 2, 3
    dilation = 1, 1
    groups = 1
    H_in = 32
    W_in = 48
    x = torch.randn(batch_size, in_channels, H_in, W_in, device='cuda',
        dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size[0
        ], kernel_size[1], device='cuda', dtype=torch.float32)
    bias = None
    out_kernel = cuda_kernel_module.forward(x, weight, bias, list(stride),
        list(padding), list(dilation), groups)
    out_ref = ref_conv2d(x, weight, bias, stride, padding, dilation, groups)
    max_diff = (out_kernel - out_ref).abs().max().item()
    assert math.isclose(max_diff, 0.0, abs_tol=0.001
        ), f'Issue with shared memory usage: max difference {max_diff} exceeds tolerance'


def test_shared_memory_allocation(cuda_kernel_module):
    batch_size = 1
    in_channels = 1
    out_channels = 1
    kernel_size = 3, 3
    stride = 1, 1
    padding = 1, 1
    dilation = 1, 1
    groups = 1
    H_in = 128
    W_in = 128
    x = torch.randn(batch_size, in_channels, H_in, W_in, device='cuda',
        dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size[0],
        kernel_size[1], device='cuda', dtype=torch.float32)
    bias = None
    out_kernel = cuda_kernel_module.forward(x, weight, bias, list(stride),
        list(padding), list(dilation), groups)
    torch.cuda.synchronize()
    out_ref = ref_conv2d(x, weight, bias, stride, padding, dilation, groups)
    max_diff = (out_kernel - out_ref).abs().max().item()
    assert math.isclose(max_diff, 0.0, abs_tol=0.001
        ), f'Shared memory allocation issue: max diff {max_diff} exceeds tolerance'
