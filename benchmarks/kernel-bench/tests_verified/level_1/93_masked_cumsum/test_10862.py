import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load

def test_invalid_dimension():
    device = torch.device('cuda')
    N, L = 128, 4000
    x = torch.randn(N, L, device=device, dtype=torch.float32)
    mask = torch.randint(0, 2, (N, L), device=device, dtype=torch.bool)
    module = build_kernel()
    with pytest.raises(Exception):
        module.forward(x, mask, 2)


if __name__ == '__main__':
    pytest.main([__file__])
