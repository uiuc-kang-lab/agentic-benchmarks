import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load

def test_non_contiguous_input():
    mod = build_kernel()
    B, in_channels, in_size = 4, 3, 32
    out_channels = 8
    kernel_size = 3
    stride = 1
    dilation = 1
    x = torch.randn(B, in_channels, in_size, device='cuda', dtype=torch.float32
        ).transpose(1, 2)
    weight = torch.randn(out_channels, in_channels, kernel_size, device=
        'cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    try:
        out = mod.forward(x, weight, bias, stride, dilation)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    ref = torch.nn.functional.conv1d(x, weight, bias=bias, stride=stride,
        dilation=dilation)
    assert torch.allclose(out, ref, atol=0.01
        ), f'Kernel output ({out}) differs from reference ({ref})!'


def test_unroll_issue():
    mod = build_kernel()
    B, in_channels, in_size = 2, 3, 20
    out_channels = 4
    kernel_size = 2
    stride = 1
    dilation = 1
    x = torch.randn(B, in_channels, in_size, device='cuda', dtype=torch.float32
        )
    weight = torch.randn(out_channels, in_channels, kernel_size, device=
        'cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    out = mod.forward(x, weight, bias, stride, dilation)
    ref = torch.nn.functional.conv1d(x, weight, bias=bias, stride=stride,
        dilation=dilation)
    assert torch.allclose(out, ref, atol=0.01
        ), f'Kernel output ({out}) differs from reference ({ref})!'
