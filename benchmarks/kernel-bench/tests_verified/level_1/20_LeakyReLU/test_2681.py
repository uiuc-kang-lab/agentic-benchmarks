import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load


@pytest.mark.skipif(not torch.cuda.is_available(), reason=
    'CUDA is not available')
def test_duplicate_processing():
    n = 1280
    x = torch.randn(n, device='cuda', dtype=torch.float32)
    negative_slope = 0.01
    ref = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    cuda_module = build_kernel()
    out = cuda_module.forward(x, negative_slope)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=0.01
        ), 'The kernel output unexpectedly matches the reference; duplicate processing issue not triggered.'
