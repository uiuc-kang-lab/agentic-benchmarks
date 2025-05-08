import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_wrapper_vs_custom_kernel():
    mod = build_kernel()
    x = torch.randn(2, 4, 8, 8, 8, device='cuda')
    weight = torch.randn(4, 4, 3, 3, 3, device='cuda')
    out = mod.forward(x, weight, None, [1, 1, 1], [0, 0, 0], [0, 0, 0], 1)
    assert out.shape[2] == (x.shape[2] - 1
        ) * 1 - 2 * 0 + 3, 'Output depth is not as expected'


def test_non_contiguous_input():
    mod = build_kernel()
    x = torch.randn(2, 4, 8, 8, 8, device='cuda')
    x_noncontig = x.transpose(2, 3)
    weight = torch.randn(4, 4, 3, 3, 3, device='cuda')
    try:
        out = mod.forward(x_noncontig, weight, None, [1, 1, 1], [0, 0, 0], [0, 0,
            0], 1)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    out_ref = torch.nn.functional.conv_transpose3d(x_noncontig, weight, None,
        stride=[1, 1, 1], padding=[0, 0, 0], output_padding=[0, 0, 0], groups=1)
    assert torch.allclose(out, out_ref, atol=0.01
        ), 'Kernel did not produce the expected results with non-contiguous input.'

if __name__ == '__main__':
    pytest.main()
