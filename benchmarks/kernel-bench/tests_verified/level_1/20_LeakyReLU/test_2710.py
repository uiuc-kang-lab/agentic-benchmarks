import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import os
import pytest
import torch
from torch.utils.cpp_extension import load

def test_non_contiguous_tensor():
    x = torch.randn(1024, device='cuda', dtype=torch.float32).view(32, 32).t()
    mod = build_kernel()
    try:
        result = mod.forward(x, 0.01)
    except RuntimeError:
        print("RuntimeError: Kernel does not support non-contiguous inputs.")
        return
    result_ref = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
    assert torch.allclose(result, result_ref
        ), 'Kernel incorrectly processed a non-contiguous tensor.'
