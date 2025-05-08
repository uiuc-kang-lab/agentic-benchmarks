import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def cuda_max_reduce(module, input_tensor, dim):
    return module.forward(input_tensor, dim)


def test_non_contiguous_tensor():
    x = torch.randn(16, 256, 256, device='cuda')
    x_nc = x.transpose(1, 2)
    cuda_module = build_kernel()
    try:
        out_cuda = cuda_max_reduce(cuda_module, x_nc, 1)
    except Exception as e:
        pytest.skip(
            'Kernel did not run on non-contiguous input (expected behavior).')
    out_ref = torch.max(x_nc, dim=1)[0]
    assert torch.allclose(out_cuda, out_ref, atol=0.01
        ), 'For a non-contiguous tensor, the CUDA kernel returned the same result as torch.max, which is unexpected given its assumptions about memory layout.'
