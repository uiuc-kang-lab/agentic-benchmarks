import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_noncontiguous_tensor():
    x = torch.randn(4, 5, 6, device='cuda', dtype=torch.float32).transpose(0, 1
        )
    ref = torch.argmin(x, dim=1)
    my_module = build_kernel()
    try:
        out = my_module.forward(x, 1)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    assert not torch.equal(out, ref
        ), 'Kernel produced correct result on non-contiguous tensor unexpectedly.'


def test_empty_reduction_dimension():
    x = torch.empty(5, 0, 7, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    with pytest.raises(Exception):
        _ = my_module.forward(x, 1)
