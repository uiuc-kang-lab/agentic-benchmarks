import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import time
import torch
from torch.utils.cpp_extension import load


def test_error_synchronization():
    M, K, N = 1024, 4096, 2048
    A = torch.randn(K + 1, M, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
