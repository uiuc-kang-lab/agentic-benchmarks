import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_kernel_launch_failure():
    module = build_kernel()
    x = torch.empty(0, dtype=torch.float32, device='cuda')
    y = module.forward(x)
    assert y.numel() == 0


def test_deprecated_dtype_usage_warning():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    module = build_kernel()
    x = torch.randn(1024, dtype=torch.float32, device='cuda')
    y = module.forward(x)
    y_ref = torch.tanh(x)
    torch.cuda.synchronize()
    assert torch.allclose(y, y_ref, atol=0.01 
        ), f'Kernel produced incorrect result for deprecated dtype. Max diff: {(y - y_ref).abs().max().item()}'
