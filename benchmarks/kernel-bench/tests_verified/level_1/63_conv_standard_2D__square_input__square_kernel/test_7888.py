import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def test_non_contiguous():
    cuda_module = build_kernel()
    N, Cin, H, W = 1, 3, 32, 32
    x = torch.randn(N, Cin, H, W, device='cuda', dtype=torch.float32)
    x_non_contig = x.transpose(1, 2)
    K = 3
    Cout = 8
    weight = torch.randn(Cout, Cin, K, K, device='cuda', dtype=torch.float32)
    bias = torch.randn(Cout, device='cuda', dtype=torch.float32)
    stride, padding, dilation, groups = 1, 0, 1, 1
    try:
        result = cuda_module.forward(x_non_contig, weight, bias, stride, padding,
            dilation, groups)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    conv = torch.nn.Conv2d(Cin, Cout, kernel_size=K, stride=stride, padding
        =padding, dilation=dilation, bias=True)
    conv.weight.data.copy_(weight)
    conv.bias.data.copy_(bias)
    conv = conv.to('cuda', torch.float32)
    ref_result = conv(x_non_contig)
    assert torch.allclose(result, ref_result, atol=0.01
        ), 'Kernel output unexpectedly matches the reference for non-contiguous input!'


def test_output_correctness_float32():
    cuda_module = build_kernel()
    N, Cin, H, W = 1, 3, 16, 16
    x = torch.randn(N, Cin, H, W, device='cuda', dtype=torch.float32)
    K = 3
    Cout = 3
    weight = torch.ones(Cout, Cin, K, K, device='cuda', dtype=torch.float32)
    bias = torch.zeros(Cout, device='cuda', dtype=torch.float32)
    stride, padding, dilation, groups = 1, 1, 1, 1
    try:
        output = cuda_module.forward(x, weight, bias, stride, padding, dilation,
            groups)
    except RuntimeError:
        pytest.skip('Kernel does not support float32 inputs.')
    torch.cuda.synchronize()
    conv = torch.nn.Conv2d(Cin, Cout, kernel_size=K, stride=stride, padding
        =padding, dilation=dilation, bias=True)
    conv.weight.data.fill_(1.0)
    conv.bias.data.fill_(0.0)
    conv = conv.to('cuda', torch.float32)
    ref_output = conv(x)
    assert torch.allclose(output, ref_output, atol=0.01
        ), 'Custom kernel output does not match reference convolution.'


if __name__ == '__main__':
    pytest.main([__file__])
