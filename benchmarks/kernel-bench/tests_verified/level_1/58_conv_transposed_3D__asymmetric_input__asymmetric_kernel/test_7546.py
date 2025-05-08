import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch import nn
from torch.utils.cpp_extension import load

def test_large_input_correctness():
    cuda_module = build_kernel()
    batch_size = 16
    in_channels = 32
    out_channels = 16
    in_depth = 16
    in_height = 32
    in_width = 64
    kT, kH, kW = 3, 5, 7
    input_tensor = torch.randn(batch_size, in_channels, in_depth, in_height,
        in_width, device='cuda', dtype=torch.float32)
    weight_tensor = torch.randn(in_channels, out_channels // 1, kT, kH, kW,
        device='cuda', dtype=torch.float32)
    bias_tensor = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    stride = [1, 1, 1]
    padding = [0, 0, 0]
    output_padding = [0, 0, 0]
    groups = 1
    native_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size
        =(kT, kH, kW), stride=tuple(stride), padding=tuple(padding),
        output_padding=tuple(output_padding), groups=groups, bias=True).cuda()
    with torch.no_grad():
        native_conv.weight.copy_(weight_tensor)
        native_conv.bias.copy_(bias_tensor)
    out_kernel = cuda_module.forward(input_tensor, weight_tensor,
        bias_tensor, stride, padding, output_padding, groups)
    out_native = native_conv(input_tensor)
    assert torch.allclose(out_kernel, out_native, atol=0.01
        ), 'Kernel output does not match native output.'
