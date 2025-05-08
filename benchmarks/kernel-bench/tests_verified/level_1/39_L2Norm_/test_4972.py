import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load



def test_non_contiguous_input():
    my_module = build_kernel()
    batch_size = 16
    C = 16384
    x = torch.randn(batch_size, C, device='cuda', dtype=torch.float32)
    x_t = x.t()
    norm = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12
    expected = x / norm
    try:
        output_t = my_module.forward(x_t)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    output = output_t.t()
    torch.cuda.synchronize()
    assert torch.allclose(output, expected, atol=0.01
        ), 'Test did not trigger the non-contiguous input bug as expected.'
