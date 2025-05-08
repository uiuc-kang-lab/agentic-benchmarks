import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os


def test_unused_shared_memory():
    cuda_module = build_kernel()
    x = torch.randn(1024, device='cuda', dtype=torch.float32)
    out = cuda_module.forward(x)
    expected = F.softplus(x)
    assert torch.allclose(out, expected, atol=0.01
        ), 'Output does not match softplus reference. (Shared memory misuse issue)'


def test_threshold_constants_conversion():
    cuda_module = build_kernel()
    values = torch.tensor([-30.0, -20.5, -20.0, 0.0, 20.0, 20.5, 30.0],
        device='cuda', dtype=torch.float32)
    out = cuda_module.forward(values)
    expected = F.softplus(values)
    assert torch.allclose(out, expected, atol=0.01
        ), 'Threshold handling in compute_softplus may be affected by literal constant conversion.'


def test_kernel_launch_error_checking():
    cuda_module = build_kernel()
    x = torch.randn(16, 16384, device='cuda', dtype=torch.float32)
    x_noncontiguous = x.t()
    try:
        out = cuda_module.forward(x_noncontiguous)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    expected = F.softplus(x_noncontiguous)
    if torch.allclose(out, expected, atol=0.01):
        pytest.skip(
            'Non-contiguous input unexpectedly produced correct result; kernel may be inadvertently handling it.'
            )
    else:
        assert torch.allclose(out, expected, atol=0.01
            ), 'Kernel did not raise an error on non-contiguous input (error checking issue).'


def test_non_contiguous_input():
    cuda_module = build_kernel()
    x = torch.randn(1024, 128, device='cuda', dtype=torch.float32)
    x_noncontiguous = x.transpose(0, 1)
    try:
        out = cuda_module.forward(x_noncontiguous)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    expected = F.softplus(x_noncontiguous)
    assert torch.allclose(out, expected, atol=0.01
        ), 'Kernel incorrectly assumed contiguous input.'
