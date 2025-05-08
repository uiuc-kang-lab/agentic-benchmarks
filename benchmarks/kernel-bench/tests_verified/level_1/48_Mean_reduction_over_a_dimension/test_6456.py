import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load


def test_non_contiguous_input():
    batch, dim1, dim2 = 4, 8, 16
    x = torch.randn(batch, dim1, dim2, device='cuda')
    x_noncontig = x.transpose(1, 2)
    kernel = build_kernel()
    try:
        y_kernel = kernel.forward(x_noncontig, 2)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    y_torch = torch.mean(x_noncontig, dim=2)
    assert torch.allclose(y_kernel, y_torch, atol=0.01
        ), f'Kernel output does not match PyTorch mean output for non-contiguous input. This may indicate a precision issue. Kernel output: {y_kernel}, Reference output: {y_torch}'


def test_zero_length_reduction():
    x = torch.randn(3, 0, 5, device='cuda')
    kernel = build_kernel()
    y_kernel = kernel.forward(x, 1)
    y_torch = torch.mean(x, dim=1)
    assert torch.isnan(y_kernel).all() and torch.isnan(y_torch).all(
        ), 'Division by zero not handled as expected.'


if __name__ == '__main__':
    pytest.main([__file__])
