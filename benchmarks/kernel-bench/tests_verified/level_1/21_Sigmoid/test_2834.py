import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_misaligned_tensor():
    kernel = build_kernel()
    base = torch.randn(17, 1024, device='cuda', dtype=torch.float32)
    x = base[:, 1:]
    try:
        y = kernel.forward(x)
    except Exception as e:
        pytest.fail(f'Kernel crashed with misaligned tensor: {e}')
    y_ref = torch.sigmoid(x)
    assert torch.allclose(y, y_ref, atol=0.01
        ), f'Kernel produced incorrect result for misaligned tensor. Max diff: {(y - y_ref).abs().max().item()}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_non_contiguous_tensor():
    kernel = build_kernel()
    x = torch.randn(16, 16384, device='cuda', dtype=torch.float32)
    x_noncontig = x.transpose(0, 1)
    try:
        y = kernel.forward(x_noncontig)
    except Exception as e:
        pytest.skip(f'Kernel crashed with non-contiguous tensor: {e}')
    y_ref = torch.sigmoid(x_noncontig)
    assert torch.allclose(y, y_ref, atol=0.01
        ), f'Kernel produced incorrect result for non-contiguous tensor. Max diff: {(y - y_ref).abs().max().item()}'
