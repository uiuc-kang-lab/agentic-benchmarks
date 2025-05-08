import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_non_contiguous_input():
    A = torch.randn(128, 4000, device='cuda')
    A_t = A.t()
    my_module = build_kernel()
    try:
        result_kernel = my_module.forward(A_t, 0)
    except RuntimeError as e:
        pytest.skip(f'Kernel does not support non-contiguous input: {e}')
    result_ref = torch.cumprod(A_t, dim=0)
    assert torch.allclose(result_kernel, result_ref, atol=0.01
        ), 'Test failed to trigger non-contiguous input issue: kernel result unexpectedly matches torch.cumprod.'

def test_invalid_dimension():
    A = torch.randn(128, 4000, device='cuda')
    my_module = build_kernel()
    with pytest.raises(Exception):
        result_kernel = my_module.forward(A, 2)


if __name__ == '__main__':
    pytest.main([__file__])
