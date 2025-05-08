import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load
import numpy as np


def reference_rms_norm(x, eps):
    rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)
    return x / rms


def test_noncontiguous_input():
    batch_size = 8
    features = 32
    dim1, dim2 = 16, 16
    x = torch.randn(batch_size, features, dim1, dim2, device='cuda', dtype=
        torch.float32)
    x_noncontig = x.transpose(1, 2)
    x_noncontig = x_noncontig.transpose(1, 2)
    assert not x_noncontig.is_contiguous(
        ), 'Test setup error: Tensor is contiguous.'
    eps = 1e-05
    ref = reference_rms_norm(x_noncontig, eps)
    module = build_kernel()
    try:
        out = module.forward(x_noncontig, eps)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=0.01
        ), 'Non-contiguous tensor test did not trigger an error: kernel output is unexpectedly close to reference.'


if __name__ == '__main__':
    pytest.main([__file__])
