import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_non_contiguous_tensor():
    x = torch.randn(16, 256, 256, device='cuda')
    x_non_contig = x.transpose(0, 1)
    my_module = build_kernel()
    try:
        out = my_module.forward(x_non_contig, 1)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    ref = torch.min(x_non_contig, dim=1)[0]
    torch.cuda.synchronize()
    assert not torch.equal(out, ref), 'Kernel produced correct result on non-contiguous tensor unexpectedly.'

def test_empty_reduction_dimension():
    x = torch.randn(16, 0, 256, device='cuda')
    my_module = build_kernel()
    with pytest.raises(Exception):
        _ = my_module.forward(x, 1)
        torch.cuda.synchronize()
