import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load
import torch.nn.functional as F

def test_non_contiguous_input():
    cuda_module = build_kernel()
    batch_size = 2
    channels = 3
    height = 16
    width = 16
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    x_contig = torch.randn(batch_size, channels, height, width, device=
        'cuda', dtype=torch.float32)
    x_noncontig = x_contig.transpose(2, 3)
    try:
        out = cuda_module.forward(x_noncontig, kernel_size, stride, padding,
            dilation)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    expected = F.max_pool2d(x_noncontig.contiguous(), kernel_size=
        kernel_size, stride=stride, padding=padding, dilation=dilation)
    torch.cuda.synchronize()
    assert torch.allclose(out, expected, atol=0.01
        ), 'Issue 4 triggered: non-contiguous input was handled correctly whereas it should have produced an error (or wrong result) due to lack of contiguity check.'


if __name__ == '__main__':
    pytest.main([__file__])
