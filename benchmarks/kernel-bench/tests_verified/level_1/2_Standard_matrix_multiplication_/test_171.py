import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_kernel_launch_error():
    my_module = build_kernel()
    A = torch.randn(32, 10, dtype=torch.float32, device='cuda')
    B = torch.randn(32, 32, dtype=torch.float32, device='cuda')
    # error is expected here
    try:
        my_module.forward(A, B)
    except RuntimeError:
        # pass
        print("RuntimeError: Kernel launch error as expected.")
        return
    else:
        assert False, 'Kernel launch error not raised as expected'
