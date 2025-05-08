import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def swish_torch(x: torch.Tensor) ->torch.Tensor:
    return x * torch.sigmoid(x)

def test_non_contiguous():
    cuda_module = build_kernel()
    x = torch.randn(256, 1024, device='cuda', dtype=torch.float32)
    x_non_contig = x.t()
    assert not x_non_contig.is_contiguous(
        ), 'Test setup error: x_non_contig should be non-contiguous.'
    try:
        y_kernel = cuda_module.forward(x_non_contig)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    y_ref = swish_torch(x_non_contig)
    assert torch.allclose(y_kernel, y_ref, atol=0.01
        ), 'Kernel incorrectly handled non-contiguous input.'


def test_large_tensor_behavior():
    cuda_module = build_kernel()
    n = 1 << 20
    x = torch.randn(n, device='cuda', dtype=torch.float32)
    y_kernel = cuda_module.forward(x)
    y_ref = swish_torch(x)
    assert torch.allclose(y_kernel, y_ref, atol=0.01
        ), 'Kernel did not compute correct swish activation on a large tensor (within safe limits).'
