import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def test_noncontiguous_tensor():
    cuda_module = build_kernel()
    x = torch.randn(32, 32, device='cuda', dtype=torch.float32)
    x_noncontig = x.t()
    try:
        out = cuda_module.forward(x_noncontig, 1.0)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    ref = F.elu(x_noncontig, alpha=1.0)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=0.01
        ), 'Kernel processed non-contiguous input as if it were contiguous.'
