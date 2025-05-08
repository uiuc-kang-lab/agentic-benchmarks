import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import torch.nn.functional as F
import pytest
from torch.utils.cpp_extension import load


def conv2d_reference(x, weight, bias, stride, padding, dilation, groups):
    return F.conv2d(x, weight, bias=bias, stride=stride, padding=padding,
        dilation=dilation, groups=groups)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_kernel_size_mismatch():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 3
    out_channels = 8
    kernel_size = 5
    height, width = 32, 32
    stride = 1
    padding = 2
    x = torch.randn(batch_size, in_channels, height, width, device='cuda',
        dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size,
        kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    output_cuda = cuda_module.forward(x, weight, bias, stride, padding, 1, 1)
    output_ref = conv2d_reference(x, weight, bias, stride, padding,
        dilation=1, groups=1)
    assert torch.allclose(output_cuda, output_ref, atol=0.01
        ), f'Test failed: CUDA kernel output does not match reference. CUDA output: {output_cuda}, Reference output: {output_ref}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_thread_block_tile_mismatch():
    cuda_module = build_kernel()
    batch_size = 1
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    height, width = 40, 40
    stride = 1
    padding = 1
    x = torch.randn(batch_size, in_channels, height, width, device='cuda',
        dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size,
        kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    output_cuda = cuda_module.forward(x, weight, bias, stride, padding, 1, 1)
    output_ref = conv2d_reference(x, weight, bias, stride, padding,
        dilation=1, groups=1)
    assert torch.allclose(output_cuda, output_ref, atol=0.01
        ), 'Test failed: CUDA kernel output matches reference even though thread block tiling is misconfigured.'
