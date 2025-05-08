import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load


def maxpool_torch(input, kernel_size, stride, padding, dilation):
    pooling = torch.nn.MaxPool2d(kernel_size, stride, padding, dilation=
        dilation)
    return pooling(input)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_fixed_unroll_issue():
    device = 'cuda'
    batch_size, channels, height, width = 2, 3, 16, 16
    input = torch.randn(batch_size, channels, height, width, device=device,
        dtype=torch.float32)
    kernel_size = 5
    stride = 1
    padding = 2
    dilation = 1
    mod = build_kernel()
    output_kernel = mod.forward(input, kernel_size, stride, padding, dilation)
    output_torch = maxpool_torch(input, kernel_size, stride, padding, dilation)
    assert torch.allclose(output_kernel, output_torch, atol=0.01
        ), 'Mismatch in max pooling results for kernel_size != 2 or 3 (possible fixed unroll issue)'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_missing_error_checking_behavior():
    device = 'cuda'
    batch_size, channels, height, width = 2, 3, 16, 16
    input = torch.randn(batch_size, channels, height, width, device=device,
        dtype=torch.float32)
    kernel_size = 2
    stride = 2
    padding = -1
    dilation = 1
    mod = build_kernel()
    with pytest.raises(Exception):
        output_kernel = mod.forward(input, kernel_size, stride, padding, dilation)

