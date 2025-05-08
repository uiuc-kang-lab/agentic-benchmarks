import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_nonstandard_block_dim():
    x = torch.randn(16, 256, 256, device='cuda', dtype=torch.float32)
    kernel = build_kernel()
    out = kernel.forward(x, 1)
    expected = torch.min(x, dim=1)[0]
    assert torch.allclose(out, expected, atol=0.01
        ), f'Kernel output does not match expected output: {out} != {expected}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_overflow_in_dimension_conversion():
    kernel = build_kernel()
    dim0 = 10000
    dim1 = 2
    dim2 = 10000
    try:
        x = torch.randn(dim0, dim1, dim2, device='cuda', dtype=torch.float32)
    except RuntimeError:
        pytest.skip('Tensor too large to allocate on this device.')
    out = kernel.forward(x, 1)
    expected = torch.min(x, dim=1)[0]
    assert torch.allclose(out, expected, atol=0.01
        ), 'Potential dimension conversion overflow detected.'


if __name__ == '__main__':
    pytest.main([__file__])
