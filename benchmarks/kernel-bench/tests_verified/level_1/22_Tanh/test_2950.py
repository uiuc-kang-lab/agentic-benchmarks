import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load


def test_non_contiguous_tensor():
    my_module = build_kernel()
    x = torch.randn(16, 16384, device='cuda', dtype=torch.float32)
    x_noncontig = x.t()
    try:
        y = my_module.forward(x_noncontig)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    y_ref = torch.tanh(x_noncontig)
    assert torch.allclose(y, y_ref, atol=1e-2
        ), f'Kernel produced incorrect result for non-contiguous tensor. Max diff: {(y - y_ref).abs().max().item()}'

def test_misaligned_storage():
    my_module = build_kernel()
    x = torch.randn(16, 16384 + 1, device='cuda', dtype=torch.float32)
    x_misaligned = x[:, 1:]
    y = my_module.forward(x_misaligned)
    torch.cuda.synchronize()
    y_ref = torch.tanh(x_misaligned)
    assert torch.allclose(y, y_ref, atol=1e-2
        ), f'Kernel produced incorrect result for misaligned storage. Max diff: {(y - y_ref).abs().max().item()}'
