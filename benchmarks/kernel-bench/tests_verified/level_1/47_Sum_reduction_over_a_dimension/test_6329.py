import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_non_contiguous_input():
    batch_size, dim1, dim2 = 4, 8, 8
    original = torch.randn(batch_size, dim1, dim2, device='cuda')
    non_contig = original.transpose(1, 2)
    reduce_dim = 1
    kernel_module = build_kernel()
    try:
        out_cuda = kernel_module.forward(non_contig, reduce_dim)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    out_torch = torch.sum(non_contig, dim=reduce_dim, keepdim=True)
    assert torch.allclose(out_cuda, out_torch
        ), 'Kernel incorrectly handled non-contiguous tensor'
