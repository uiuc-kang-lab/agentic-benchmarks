import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_division_by_zero():
    my_module = build_kernel()
    x = torch.full((16, 16384), float('-inf'), device='cuda', dtype=torch.
        float32)
    y = my_module.forward(x)
    assert torch.isnan(y).any(
        ), 'Expected NaN values due to division by zero but none were found.'


def test_empty_tensor():
    my_module = build_kernel()
    x = torch.empty((0, 16384), device='cuda', dtype=torch.float32)
    y = my_module.forward(x)
    assert y.numel(
        ) == 0, 'Expected an empty output tensor for an empty input.'
