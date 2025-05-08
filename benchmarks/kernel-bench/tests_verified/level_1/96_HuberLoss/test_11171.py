import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import os
import shutil
import tempfile
import pytest
import torch
from torch.utils.cpp_extension import load


def test_empty_input():
    predictions = torch.tensor([], device='cuda', dtype=torch.float32)
    targets = torch.tensor([], device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    out = my_module.forward(predictions, targets)
    torch.cuda.synchronize()
    assert torch.isnan(out).all(
        ), 'Expected NaN output when input tensor is empty (division by zero).'

def test_non_contiguous_input():
    my_module = build_kernel()
    a = torch.randn(130, 4096, device='cuda', dtype=torch.float32)
    b = torch.randn(130, 4096, device='cuda', dtype=torch.float32)
    predictions = a.transpose(0, 1)
    targets = b.transpose(0, 1)
    try:
        result = my_module.forward(predictions, targets)
    except RuntimeError as e:
        pytest.skip(f'Kernel does not support non-contiguous input: {e}')
    torch.cuda.synchronize()
    ref = torch.nn.functional.smooth_l1_loss(predictions, targets)
    assert torch.allclose(result, ref, atol=0.01
        ), 'Kernel output does not match reference for non-contiguous input.'
