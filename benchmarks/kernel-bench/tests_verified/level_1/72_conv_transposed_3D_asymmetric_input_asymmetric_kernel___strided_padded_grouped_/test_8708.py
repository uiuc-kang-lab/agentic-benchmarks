import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load


class RefConvTranspose3d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, output_padding, groups, bias):
        super().__init__()
        self.conv_transpose3d = torch.nn.ConvTranspose3d(in_channels,
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv_transpose3d(x)


def test_race_condition():
    batch_size = 2
    in_channels = 4
    out_channels = 8
    kernel_size = 3, 3, 3
    stride = 2, 2, 2
    padding = 1, 1, 1
    output_padding = 1, 1, 1
    groups = 2
    bias_flag = True
    device = torch.device('cuda')
    torch.manual_seed(42)
    x = torch.randn(batch_size, in_channels, 8, 8, 8, device=device, dtype=
        torch.float32)
    ref_model = RefConvTranspose3d(in_channels, out_channels, kernel_size,
        stride, padding, output_padding, groups, bias_flag).to(device)
    weight = ref_model.conv_transpose3d.weight.detach().clone()
    bias_tensor = ref_model.conv_transpose3d.bias.detach().clone(
        ) if bias_flag else None
    custom_kernel = build_kernel()
    out_custom = custom_kernel.forward(x, weight, bias_tensor, [stride[0],
        stride[1], stride[2]], [padding[0], padding[1], padding[2]], [
        output_padding[0], output_padding[1], output_padding[2]], groups)
    out_ref = ref_model(x)
    diff = (out_custom - out_ref).abs().max().item()
    assert diff < 0.001, f'Expected a significant difference due to race conditions, but diff={diff}'


def test_non_multiple_of_block_size():
    batch_size = 1
    in_channels = 3
    out_channels = 6
    kernel_size = 3, 3, 3
    stride = 1, 1, 1
    padding = 1, 1, 1
    output_padding = 0, 0, 0
    groups = 1
    bias_flag = False
    device = torch.device('cuda')
    torch.manual_seed(999)
    x = torch.randn(batch_size, in_channels, 5, 7, 9, device=device, dtype=
        torch.float32)
    ref_model = RefConvTranspose3d(in_channels, out_channels, kernel_size,
        stride, padding, output_padding, groups, bias_flag).to(device)
    weight = ref_model.conv_transpose3d.weight.detach().clone()
    bias_tensor = None
    custom_kernel = build_kernel()
    out_custom = custom_kernel.forward(x, weight, bias_tensor, [stride[0],
        stride[1], stride[2]], [padding[0], padding[1], padding[2]], [
        output_padding[0], output_padding[1], output_padding[2]], groups)
    out_ref = ref_model(x)
    diff = (out_custom - out_ref).abs().max().item()
    assert diff < 0.001, f'Expected difference due to non-generalized block size handling, but diff={diff}'
