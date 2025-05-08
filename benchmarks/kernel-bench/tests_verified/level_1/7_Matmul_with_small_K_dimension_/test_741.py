import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load


def test_incorrect_matrix_dimensions():
    A = torch.randn(64, 32, device='cuda', dtype=torch.float32)
    B = torch.randn(64, 64, device='cuda', dtype=torch.float32)
    kernel_module = build_kernel()
    with pytest.raises(Exception):
        kernel_module.forward(A, B)
