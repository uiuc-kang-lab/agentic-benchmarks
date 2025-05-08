import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import os
import re
import pytest
import torch
from torch.utils.cpp_extension import load

def test_mismatched_inner_dimensions():
    mod = build_kernel()
    M, K, N = 32, 50, 40
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(N, K + 1, dtype=torch.float32, device='cuda')
    with pytest.raises(RuntimeError):
        mod.forward(A, B)
