import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import os


def test_conditional_sync():
    torch.manual_seed(0)
    x = torch.randn(4, 97, device='cuda', dtype=torch.float32).contiguous()
    module = build_kernel()
    kernel_result = module.forward(x, 1)
    ref_result = torch.cumsum(x, dim=1)
    assert torch.allclose(kernel_result, ref_result, atol=0.01
        ), 'Conditional __syncthreads usage issue not triggered: kernel output unexpectedly matches torch.cumsum.'


def test_scan_dim_not_multiple_of_32():
    torch.manual_seed(0)
    x = torch.randn(10, 45, device='cuda', dtype=torch.float32).contiguous()
    module = build_kernel()
    kernel_result = module.forward(x, 1)
    ref_result = torch.cumsum(x, dim=1)
    assert torch.allclose(kernel_result, ref_result, atol=0.01
        ), 'Work distribution issue not triggered: kernel output for non-multiple-of-32 scan dimension matches torch.cumsum.'
