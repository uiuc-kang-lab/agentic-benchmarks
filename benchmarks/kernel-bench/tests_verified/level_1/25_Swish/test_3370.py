import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
import math
from torch.utils.cpp_extension import load


def reference_swish(x: torch.Tensor) ->torch.Tensor:
    return x * (1.0 / (1.0 + torch.exp(-x)))


def test_constant_naming_issue():
    device = 'cuda'
    x = torch.randn(1024, device=device, dtype=torch.float32)
    module = build_kernel()
    y_kernel = module.forward(x)
    y_ref = reference_swish(x)
    assert torch.allclose(y_kernel, y_ref, atol=0.01
        ), f'Kernel computation with constant value error: max diff {torch.abs(y_kernel - y_ref).max()}'

def test_noncontiguous_input_issue():
    device = 'cuda'
    x = torch.randn(128, 128, device=device, dtype=torch.float32)
    x_noncontig = x.t()
    module = build_kernel()
    try:
        y_kernel = module.forward(x_noncontig)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    y_ref = reference_swish(x_noncontig)
    diff = torch.abs(y_kernel - y_ref).max().item()
    assert diff > 0.001, f'Kernel output unexpectedly matches reference for noncontiguous input; diff {diff}'


def test_kernel_launch_error_checking():
    device = 'cuda'
    x = torch.empty(0, device=device, dtype=torch.float32)
    module = build_kernel()
    y_kernel = module.forward(x)
    torch.cuda.synchronize()
    assert y_kernel.numel(
        ) == 0, 'Kernel did not correctly handle an empty input tensor.'


def test_launch_configuration_issue():
    device = 'cuda'
    n = 10 ** 6
    x = torch.randn(n, device=device, dtype=torch.float32)
    module = build_kernel()
    y_kernel = module.forward(x)
    y_ref = reference_swish(x)
    assert torch.allclose(y_kernel, y_ref, atol=0.01
        ), 'Kernel computed incorrect swish on a large tensor, possibly due to fixed launch configuration.'


if __name__ == '__main__':
    pytest.main([__file__])
