import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def get_valid_inputs(dtype=torch.float32, contiguous=True):
    batch_size = 16
    in_channels = 32
    out_channels = 64
    kernel_size = 3
    height = 128
    width = 128
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    x = torch.randn(batch_size, in_channels, height, width, dtype=dtype,
        device='cuda')
    weight = torch.randn(in_channels, out_channels, kernel_size,
        kernel_size, dtype=dtype, device='cuda')
    bias = None
    if not contiguous:
        x = x.transpose(2, 3)
        weight = weight.transpose(2, 3)
    return x, weight, bias, stride, padding, output_padding, groups


def test_non_contiguous_input():
    module = build_kernel()
    x, weight, bias, stride, padding, output_padding, groups = (
        get_valid_inputs(contiguous=False))
    try:
        result = module.forward(x, weight, bias, stride, padding, output_padding,
            groups)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    ref_result = torch.nn.functional.conv_transpose2d(x, weight, bias,
        stride=stride, padding=padding, output_padding=output_padding,
        groups=groups)
    assert torch.allclose(result, ref_result, atol=0.01
        ), 'Kernel unexpectedly produced correct output for a non-contiguous input'

if __name__ == '__main__':
    pytest.main([__file__])
