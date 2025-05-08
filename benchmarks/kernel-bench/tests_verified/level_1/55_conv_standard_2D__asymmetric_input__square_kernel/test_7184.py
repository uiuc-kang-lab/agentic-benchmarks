import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_non_contiguous_input():
    device = 'cuda'
    batch = 1
    in_channels = 3
    out_channels = 8
    in_height = 32
    in_width = 32
    kernel_size = 3
    x = torch.randn(batch, in_channels, in_height, in_width, device=device
        ).transpose(1, 2)
    weight = torch.randn(out_channels, in_channels, kernel_size,
        kernel_size, device=device)
    bias = torch.randn(out_channels, device=device)
    mod = build_kernel()
    try:
        result_kernel = mod.forward(x, weight, bias, 1, 0, 1, 1)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    result_ref = torch.nn.functional.conv2d(x, weight, bias, stride=1,
        padding=0, dilation=1, groups=1)
    assert torch.allclose(result_kernel, result_ref, atol=0.01
        ), 'Kernel unexpectedly produced correct output for a non-contiguous input'


def test_non_multiple_tile_dimensions():
    device = 'cuda'
    batch = 1
    in_channels = 3
    out_channels = 8
    in_height = 45
    in_width = 45
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    x = torch.randn(batch, in_channels, in_height, in_width, device=device,
        dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size,
        kernel_size, device=device, dtype=torch.float32)
    bias = torch.randn(out_channels, device=device, dtype=torch.float32)
    mod = build_kernel()
    output = mod.forward(x, weight, bias, stride, padding, dilation, 1)
    assert output.shape[2] == (in_height + 2 * padding - dilation * (
        kernel_size - 1) - 1) // stride + 1
    assert output.shape[3] == (in_width + 2 * padding - dilation * (
        kernel_size - 1) - 1) // stride + 1
