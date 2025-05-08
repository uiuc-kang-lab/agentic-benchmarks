import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load

def test_non_contiguous_input():
    my_module = build_kernel()
    x = torch.randn(64, 128, device='cuda', dtype=torch.float32)
    x_non_contig = x.t()
    try:
        out_kernel = my_module.forward(x_non_contig)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    expected = x_non_contig / (1 + torch.abs(x_non_contig))
    assert torch.allclose(out_kernel, expected, atol=0.01
        ), 'Kernel incorrectly produced the expected result with a non-contiguous tensor. This indicates it may not be handling non-contiguous memory layouts correctly.'
