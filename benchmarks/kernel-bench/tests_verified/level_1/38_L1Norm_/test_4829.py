import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import numpy as np


def torch_l1_norm(x: torch.Tensor, dim: int=1):
    s = torch.sum(torch.abs(x), dim=dim, keepdim=True)
    return x / s

def test_kernel_zero_row_behavior():
    cuda_module = build_kernel()
    x = torch.randn(16, 16384, device='cuda', dtype=torch.float32)
    x[0] = 0.0
    out_kernel = cuda_module.forward(x)
    out_ref = torch_l1_norm(x, dim=1)
    kernel_row = out_kernel[0]
    ref_row = out_ref[0]
    assert torch.isnan(ref_row).all(
        ), 'Reference computation for zero row should result in NaNs.'
    assert torch.equal(kernel_row, torch.zeros_like(kernel_row)
        ), 'Kernel did not produce zeros on zero input row.'
