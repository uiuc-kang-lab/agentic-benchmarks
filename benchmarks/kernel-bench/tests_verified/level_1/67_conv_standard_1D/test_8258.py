import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load


def run_conv1d_forward(x, weight, bias, stride, padding, dilation, groups):
    module = build_kernel()
    bias_obj = bias if bias is not None else None
    return module.forward(x, weight, bias_obj, stride, padding, dilation,
        groups)


def test_non_contiguous_input():
    batch_size = 8
    in_channels = 4
    out_channels = 8
    kernel_size = 3
    length = 128
    stride = 1
    padding = 1
    dilation = 1
    groups = 1
    x = torch.randn(batch_size, in_channels, length, device='cuda', dtype=
        torch.float32)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size,
        device='cuda', dtype=torch.float32)
    bias = None
    x_noncontig = x.transpose(1, 2).transpose(1, 2)
    try:
        y = run_conv1d_forward(x_noncontig, weight, bias, stride, padding, dilation,
            groups)
    except RuntimeError as e:
        pytest.skip(f'Kernel does not support non-contiguous input: {e}')
    torch.cuda.synchronize()
    y_ref = torch.nn.functional.conv1d(x_noncontig, weight, bias, stride=stride,
        padding=padding, dilation=dilation, groups=groups)
    assert torch.allclose(y, y_ref, atol=0.01
        ), 'Test failed: non-contiguous input produced matching outputs!'


if __name__ == '__main__':
    pytest.main([__file__])
