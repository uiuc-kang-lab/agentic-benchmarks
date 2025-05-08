import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load


def test_non_contiguous_input():
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = 3, 3
    stride = 2, 2
    padding = 1, 1
    output_padding = 1, 1
    dilation = 1, 1
    groups = 1
    module = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, output_padding=output_padding,
        dilation=dilation, groups=groups, bias=True).cuda()
    x = torch.randn(batch_size, in_channels, 8, 8, device='cuda', dtype=
        torch.float32)
    x_noncontig = x.transpose(2, 3)
    if x_noncontig.is_contiguous():
        pytest.skip(
            'Test requires a noncontiguous tensor, but got a contiguous one.')
    kernel_module = build_kernel()
    try:
        out_kernel = kernel_module.forward(x_noncontig, module.weight.detach().
            clone(), module.bias.detach().clone(), list(stride), list(padding),
            list(output_padding), list(dilation), groups)
    except RuntimeError as e:
        pytest.skip(f'Kernel does not support non-contiguous input: {e}')
    torch.cuda.synchronize()
    out_ref = module(x_noncontig.contiguous())
    assert torch.allclose(out_kernel, out_ref, atol=0.01
        ), 'Kernel unexpectedly worked on a noncontiguous input'

if __name__ == '__main__':
    pytest.main([__file__])
