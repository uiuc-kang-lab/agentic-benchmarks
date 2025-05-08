import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_count_exclude_pad_issue():
    kernel_module = build_kernel()
    N, C, H, W = 1, 1, 4, 4
    x = torch.ones(N, C, H, W, dtype=torch.float32, device='cuda')
    kernel_size = 3
    stride = 1
    padding = 1
    out_kernel = kernel_module.forward(x, kernel_size, stride, padding)
    torch.cuda.synchronize()
    outH = (H + 2 * padding - kernel_size) // stride + 1
    outW = (W + 2 * padding - kernel_size) // stride + 1
    out_ref = torch.nn.AvgPool2d(kernel_size, stride, padding)(x)
    if torch.allclose(out_kernel, out_ref, atol=0.01):
        raise AssertionError(
            'Kernel averaging incorrectly matches count_exclude_pad behavior.')

