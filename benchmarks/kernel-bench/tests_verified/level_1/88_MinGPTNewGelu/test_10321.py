import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import math

def gelu_reference(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch
        .pow(x, 3))))


@pytest.fixture(scope='module')
def kernel_module():
    return build_kernel()

def test_non_contiguous(kernel_module):
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
    x_non_contig = x.t()
    try:
        y = kernel_module.forward(x_non_contig)
    except RuntimeError as e:
        pytest.skip(f'Kernel does not support non-contiguous tensor: {e}')
    y_ref = gelu_reference(x_non_contig)
    y_ref = y_ref.contiguous()
    assert torch.allclose(y, y_ref, atol=0.001
        ), 'Kernel incorrectly handled non-contiguous input. The output should be identical to the reference implementation.'


def test_fixed_tiling(kernel_module):
    total_elements = 12345
    x = torch.randn(total_elements, device='cuda', dtype=torch.float32)
    y = kernel_module.forward(x)
    y_ref = gelu_reference(x)
    assert torch.allclose(y, y_ref, atol=0.01
        ), 'Kernel output is incorrect for an input size not divisible by the tile size. The tiling/unrolling logic may need to be more general.'
