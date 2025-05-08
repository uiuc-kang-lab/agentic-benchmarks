import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load

def test_non_contiguous_tensor():
    my_module = build_kernel()
    x = torch.randn(1024, 32, device='cuda', dtype=torch.float32)
    non_contiguous_x = x.t()
    try:
        returned = my_module.forward(non_contiguous_x)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    expected = non_contiguous_x / (1 + torch.abs(non_contiguous_x))
    torch.cuda.synchronize()
    assert torch.allclose(returned, expected, atol=0.01
        ), 'Kernel did not handle non-contiguous tensor correctly.'


def test_empty_tensor():
    my_module = build_kernel()
    x = torch.empty(0, device='cuda', dtype=torch.float32)
    returned = my_module.forward(x)
    torch.cuda.synchronize()
    assert returned.numel(
        ) == 0, 'Kernel did not properly handle an empty tensor input.'


if __name__ == '__main__':
    pytest.main([__file__])
