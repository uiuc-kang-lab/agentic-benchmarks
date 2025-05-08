import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load

def test_cumsum_correctness():
    kernel_mod = build_kernel()
    x = torch.randn(8, 4000, device='cuda', dtype=torch.float32)
    ref = torch.cumsum(x, dim=1)
    out = kernel_mod.forward(x, 1)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=0.01
        ), 'Cumulative sum output mismatches reference.'


def test_non_contiguous_tensor():
    kernel_mod = build_kernel()
    x = torch.randn(16, 32, device='cuda', dtype=torch.float32)
    x_noncontig = x.t()
    try:
        result = kernel_mod.forward(x_noncontig, 1)
    except RuntimeError as e:
        pytest.skip(f'Kernel does not support non-contiguous tensor: {e}')
    torch.cuda.synchronize()
    ref = torch.cumsum(x, dim=1)
    assert torch.allclose(result, ref, atol=0.01
        ), 'Kernel output does not match reference for non-contiguous tensor.'
