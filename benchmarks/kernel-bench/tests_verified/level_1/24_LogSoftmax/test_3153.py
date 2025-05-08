import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load

def test_empty_reduction_dimension():
    kernel_module = build_kernel()
    batch_size = 4
    dim = 0
    x = torch.randn(batch_size, dim, device='cuda')
    y_ref = torch.log_softmax(x, dim=1)
    y_kernel = kernel_module.forward(x, 1)
    assert y_kernel.shape == y_ref.shape, 'Output shape mismatch for empty dim.'
    assert torch.allclose(y_kernel, y_ref
        ), 'Kernel output unexpectedly matches the reference in the empty-dim case.'


def test_invalid_dimension():
    kernel_module = build_kernel()
    batch_size = 4
    dim = 1024
    x = torch.randn(batch_size, dim, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _ = kernel_module.forward(x, 5)
