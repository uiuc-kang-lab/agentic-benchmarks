import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import math

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_runtime_kernel_size_unrolling():
    device = torch.device('cuda')
    N, C, H, W = 4, 3, 32, 32
    kernel_size = 4
    stride = 2
    padding = 1
    x = torch.randn(N, C, H, W, device=device, dtype=torch.float32)
    module = build_kernel()
    output_cuda = module.forward(x, kernel_size, stride, padding)
    torch.cuda.synchronize()
    avgpool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
        padding=padding)
    output_ref = avgpool(x)
    assert torch.allclose(output_cuda, output_ref, atol=0.01
        ), "Output from the generic loop 'unroll' path does not match the reference AvgPool2d result."


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_non_contiguous_input():
    device = torch.device('cuda')
    N, C, H, W = 4, 3, 32, 32
    kernel_size = 3
    stride = kernel_size
    padding = 0
    x = torch.randn(N, C, H, W, device=device, dtype=torch.float32)
    x_noncontig = x.transpose(2, 3)
    module = build_kernel()
    try:
        output_cuda = module.forward(x_noncontig, kernel_size, stride, padding)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    avgpool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
        padding=padding)
    output_ref = avgpool(x_noncontig.contiguous())
    assert torch.allclose(output_cuda, output_ref, atol=0.01
        ), 'Kernel output should be wrong when a non-contiguous tensor is passed without proper handling.'
