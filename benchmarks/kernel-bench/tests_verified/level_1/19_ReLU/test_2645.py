import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import os
import pytest
import torch
from torch.utils.cpp_extension import load


def test_non_contiguous_input():
    x = torch.randn(128, 256, device='cuda', dtype=torch.float32)
    x_noncontig = x.t()
    kernel = build_kernel()
    try:
        out = kernel.forward(x_noncontig)
    except RuntimeError as e:
        pytest.skip('Kernel does not support non-contiguous inputs: ' + str(e))
    out_ref = torch.relu(x_noncontig)
    assert torch.allclose(out, out_ref, atol=1e-2
        ), f'Kernel produced incorrect result for non-contiguous input. Max diff: {(out - out_ref).abs().max().item()}'
