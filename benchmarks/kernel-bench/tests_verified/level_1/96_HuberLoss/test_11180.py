import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load
import os


def test_misaligned_memory():
    my_module = build_kernel()
    N = 4097
    base = torch.randn(128, N + 1, device='cuda', dtype=torch.float32)
    predictions = base[:, 1:]
    targets = base[:, :-1]
    expected = torch.nn.functional.smooth_l1_loss(predictions, targets)
    result = my_module.forward(predictions, targets)
    torch.cuda.synchronize()
    assert torch.allclose(result, expected, atol=0.01
        ), 'Test misaligned_memory: Kernel unexpectedly matched the expected output despite misalignment.'


def test_non_multiple_four():
    my_module = build_kernel()
    predictions = torch.randn(128, 4097, device='cuda', dtype=torch.float32)
    targets = torch.randn(128, 4097, device='cuda', dtype=torch.float32)
    expected = torch.nn.functional.smooth_l1_loss(predictions, targets)
    result = my_module.forward(predictions, targets)
    torch.cuda.synchronize()
    assert torch.allclose(result, expected, atol=0.01
        ), 'Test non_multiple_four: Kernel unexpectedly computed correct results when input size is not a multiple of 4.'


def test_warp_reduction():
    my_module = build_kernel()
    predictions = torch.randn(130, device='cuda', dtype=torch.float32)
    targets = torch.randn(130, device='cuda', dtype=torch.float32)
    expected = torch.nn.functional.smooth_l1_loss(predictions, targets)
    result = my_module.forward(predictions, targets)
    torch.cuda.synchronize()
    assert torch.allclose(result, expected, atol=0.01
        ), 'Test warp_reduction: Kernel unexpectedly produced correct reduction results.'
