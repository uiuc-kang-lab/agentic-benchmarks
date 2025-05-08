import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_incompatible_dimensions():
    my_module = build_kernel()
    A = torch.randn(32, 16, dtype=torch.float32, device='cuda')
    B = torch.randn(32, 32, dtype=torch.float32, device='cuda')
    with pytest.raises(Exception):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()


if __name__ == '__main__':
    pytest.main([__file__])
