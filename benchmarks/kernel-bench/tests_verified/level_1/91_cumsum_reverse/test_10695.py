import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load


def test_race_condition_in_constant_memory():
    torch.manual_seed(0)
    batch_size = 16
    n = 512
    x = torch.randn(batch_size, n, device='cuda', dtype=torch.float32)
    x_expected = torch.cumsum(x.flip(-1), dim=-1).flip(-1)
    cuda_module = build_kernel()
    x_result = cuda_module.forward(x, 1)
    torch.cuda.synchronize()
    assert torch.allclose(x_result, x_expected, atol=0.01
        ), 'Race condition issue not triggered: kernel result unexpectedly matches expected output.'
