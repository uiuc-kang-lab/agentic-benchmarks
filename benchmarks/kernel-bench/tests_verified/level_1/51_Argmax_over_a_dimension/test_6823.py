import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import os


def test_invalid_dim():
    my_module = build_kernel()
    x = torch.randn(16, 256, 256, dtype=torch.float32, device='cuda')
    invalid_dim = 5
    with pytest.raises(Exception):
        my_module.forward(x, invalid_dim)


def test_reduction_edge_case():
    my_module = build_kernel()
    x = torch.randn(16, 1, 256, dtype=torch.float32, device='cuda')
    indices = my_module.forward(x, 1)
    expected = torch.zeros([16, 256], dtype=torch.long, device='cuda')
    assert torch.equal(indices, expected
        ), f'Reduction failed: got {indices}, expected {expected}'
