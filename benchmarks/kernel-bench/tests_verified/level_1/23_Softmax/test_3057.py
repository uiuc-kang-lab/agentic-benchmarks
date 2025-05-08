import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import math
import pytest
import torch
from torch.utils.cpp_extension import load


def test_all_minus_infinity():
    my_module = build_kernel()
    batch_size = 4
    num_features = 100
    x = torch.randn(batch_size, num_features, dtype=torch.float32, device=
        'cuda')
    x[2, :] = float('-inf')
    y = my_module.forward(x)
    torch.cuda.synchronize()
    assert torch.isnan(y[2, :]).all(
        ), 'Expected NaNs for a row with all -infinity values.'
