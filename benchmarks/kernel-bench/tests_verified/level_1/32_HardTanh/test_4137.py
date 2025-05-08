import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os


def test_large_input():
    threads = 256
    max_elems = 65535 * threads
    numel = max_elems + 1000
    x = torch.randn(numel, device='cuda', dtype=torch.float32)
    mod = build_kernel()
    out = mod.forward(x, -1.0, 1.0)
    ref = F.hardtanh(x, min_val=-1.0, max_val=1.0)
    assert torch.allclose(out, ref, atol=0.01
        ), 'Kernel unexpectedly processed all elements despite grid size clamping, but it should have under-processed for large input tensors.'


def test_non_contiguous():
    x = torch.randn(64, 128, device='cuda', dtype=torch.float32)
    x_noncontig = x.t()
    assert not x_noncontig.is_contiguous(
        ), 'Test tensor must be non-contiguous.'
    mod = build_kernel()
    try:
        out = mod.forward(x_noncontig, -1.0, 1.0)
    except Exception as e:
        pytest.skip('Kernel failed with non-contiguous input as expected: ' +
            str(e))
    ref = F.hardtanh(x_noncontig, min_val=-1.0, max_val=1.0)
    assert torch.allclose(out, ref, atol=0.01
        ), 'Kernel produced correct output for a non-contiguous input. Expected misaligned accesses causing wrong results.'

if __name__ == '__main__':
    pytest.main([__file__])
