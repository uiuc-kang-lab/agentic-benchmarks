import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import math
import pytest
import torch
from torch.utils.cpp_extension import load


@pytest.fixture(scope='module')
def kernel_module():
    return build_kernel()

def test_non_contiguous_input(kernel_module):
    x = torch.randn(32, 64, device='cuda', dtype=torch.float32)
    x_noncontig = x.t()
    try:
        y_cuda = kernel_module.forward(x_noncontig)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    y_ref = torch.selu(x_noncontig)
    diff = (y_ref - y_cuda).abs().max().item()
    assert diff > 1e-05, f'Expected a significant difference for non-contiguous input but got {diff}'


def test_3d_thread_block_indexing_limitation(kernel_module):
    numel = 2 ** 18
    x = torch.randn(numel, device='cuda', dtype=torch.float32)
    y_ref = torch.selu(x)
    y_cuda = kernel_module.forward(x)
    assert torch.allclose(y_ref, y_cuda, atol=0.01
        ) is False, 'Expected the 2D indexing scheme to fail in a more general (3D) situation'
