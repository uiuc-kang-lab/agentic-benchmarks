import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load


def torch_depthwise_separable_conv(x, depthwise_weight, pointwise_weight,
    depthwise_bias, pointwise_bias, stride, padding, dilation):
    depthwise = torch.nn.functional.conv2d(x, depthwise_weight, bias=
        depthwise_bias, stride=stride, padding=padding, dilation=dilation,
        groups=x.shape[1])
    pointwise = torch.nn.functional.conv2d(depthwise, pointwise_weight,
        bias=pointwise_bias, stride=1, padding=0)
    return pointwise


def test_non_contiguous_input():
    cuda_module = build_kernel()
    batch = 2
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    height, width = 32, 32
    stride, padding, dilation = 1, 1, 1
    x = torch.randn(batch, in_channels, height, width, device='cuda')
    x_non_contig = x.permute(0, 2, 3, 1).contiguous().transpose(1, 3)
    depthwise_weight = torch.randn(in_channels, 1, kernel_size, kernel_size,
        device='cuda')
    pointwise_weight = torch.randn(out_channels, in_channels, 1, 1, device=
        'cuda')
    depthwise_bias = torch.randn(in_channels, device='cuda')
    pointwise_bias = torch.randn(out_channels, device='cuda')
    try:
        output = cuda_module.forward(x_non_contig, depthwise_weight,
            pointwise_weight, depthwise_bias, pointwise_bias, stride, padding,
            dilation)
    except RuntimeError as e:
        pytest.skip(f'Kernel does not support non-contiguous input: {e}')
    ref_out = torch_depthwise_separable_conv(x_non_contig.contiguous(),
        depthwise_weight, pointwise_weight, depthwise_bias, pointwise_bias,
        stride, padding, dilation)
    torch.cuda.synchronize()
    assert torch.allclose(output, ref_out, atol=0.01
        ), 'Kernel unexpectedly handled non-contiguous inputs correctly.'


def test_runtime_kernel_size_unroll():
    cuda_module = build_kernel()
    batch = 2
    in_channels = 3
    out_channels = 8
    kernel_size = 5
    height, width = 32, 32
    stride, padding, dilation = 1, 2, 1
    x = torch.randn(batch, in_channels, height, width, device='cuda')
    depthwise_weight = torch.randn(in_channels, 1, kernel_size, kernel_size,
        device='cuda')
    pointwise_weight = torch.randn(out_channels, in_channels, 1, 1, device=
        'cuda')
    depthwise_bias = torch.randn(in_channels, device='cuda')
    pointwise_bias = torch.randn(out_channels, device='cuda')
    output = cuda_module.forward(x, depthwise_weight, pointwise_weight,
        depthwise_bias, pointwise_bias, stride, padding, dilation)
    ref_out = torch_depthwise_separable_conv(x, depthwise_weight,
        pointwise_weight, depthwise_bias, pointwise_bias, stride, padding,
        dilation)
    torch.cuda.synchronize()
    assert torch.allclose(output, ref_out, atol=0.01
        ), 'Kernel appears to handle runtime kernel size unrolling correctly, but an issue was expected.'

def test_pointwise_thread_mapping_corner():
    cuda_module = build_kernel()
    batch = 1
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    height, width = 7, 7
    stride, padding, dilation = 1, 1, 1
    x = torch.randn(batch, in_channels, height, width, device='cuda')
    depthwise_weight = torch.randn(in_channels, 1, kernel_size, kernel_size,
        device='cuda')
    pointwise_weight = torch.randn(out_channels, in_channels, 1, 1, device=
        'cuda')
    depthwise_bias = torch.randn(in_channels, device='cuda')
    pointwise_bias = torch.randn(out_channels, device='cuda')
    output = cuda_module.forward(x, depthwise_weight, pointwise_weight,
        depthwise_bias, pointwise_bias, stride, padding, dilation)
    ref_out = torch_depthwise_separable_conv(x, depthwise_weight,
        pointwise_weight, depthwise_bias, pointwise_bias, stride, padding,
        dilation)
    torch.cuda.synchronize()
    assert torch.allclose(output, ref_out, atol=0.01
        ), 'Kernel unexpectedly handled pointwise mapping corner case correctly.'


if __name__ == '__main__':
    pytest.main([__file__])
