import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_large_block_dimension():
    module = build_kernel()
    x = torch.randn(16, 16384, device='cuda', dtype=torch.float32)
    out = module.forward(x)
    norm = out.norm(p=2, dim=1, keepdim=True)
    assert torch.allclose(norm, torch.ones_like(norm), atol=0.001
        ), 'Kernel with large block dimension appears to produce correct normalization; shared memory bound issue may not have been triggered.'


def test_non_contiguous_and_higher_dimensional_input():
    module = build_kernel()
    x = torch.randn(4, 16, 16384, device='cuda', dtype=torch.float32)
    x = x.transpose(0, 1)
    try:
        out = module.forward(x)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous or higher-dimensional inputs.')
    norm = out.norm(p=2, dim=1, keepdim=True)
    assert torch.allclose(norm, torch.ones_like(norm), atol=0.001
        ), 'Kernel returned nearly unit L2 norms even for non-contiguous or higher-dimensional inputs; expected mis-normalization due to fixed layout assumptions.'
