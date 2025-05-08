import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load

def test_non_contiguous_input():
    my_module = build_kernel()
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.float32).t()
    try:
        result_kernel = my_module.forward(x)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    result_torch = torch.nn.functional.softsign(x)
    assert torch.allclose(result_kernel, result_torch, atol=0.01
        ), 'Kernel output does not match expected Softsign output for non-contiguous input.'

