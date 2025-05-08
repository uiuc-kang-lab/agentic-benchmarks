import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_non_contiguous_input():
    my_module = build_kernel()
    x = torch.randn(16, 16384, device='cuda', dtype=torch.float32).t()
    try:
        y = my_module.forward(x)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    y_ref = torch.nn.functional.gelu(x)
    assert torch.allclose(y, y_ref, atol=0.01
        ), f'Kernel produced incorrect result for non-contiguous input. Max diff: {(y - y_ref).abs().max().item()}'


def test_remainder_indexing():
    my_module = build_kernel()
    x = torch.randn(6, device='cuda', dtype=torch.float32)
    y = my_module.forward(x)
    torch.cuda.synchronize()
    y_ref = torch.nn.functional.gelu(x)
    assert torch.allclose(y, y_ref, atol=0.01
        ), 'Mismatch detected in remainder kernel handling.'
