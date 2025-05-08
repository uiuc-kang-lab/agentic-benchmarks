import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_count_include_pad_issue():
    batch_size = 1
    channels = 1
    in_d = 5
    in_h = 5
    in_w = 5
    kernel_size = 3
    stride = 1
    padding = 1
    x = torch.ones(batch_size, channels, in_d, in_h, in_w, device='cuda',
        dtype=torch.float32)
    cuda_module = build_kernel()
    out_cuda = cuda_module.forward(x, kernel_size, stride, padding)
    avg_pool = torch.nn.AvgPool3d(kernel_size=kernel_size, stride=stride,
        padding=padding, count_include_pad=False)
    out_torch = avg_pool(x)
    assert torch.allclose(out_cuda, out_torch, atol=0.01
        ), 'Kernel incorrectly handles the count_include_pad option!'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_non_contiguous_input():
    batch_size = 2
    channels = 3
    in_d = 8
    in_h = 8
    in_w = 8
    kernel_size = 3
    stride = 2
    padding = 1
    x = torch.randn(batch_size, channels, in_d, in_h, in_w, device='cuda',
        dtype=torch.float32)
    x_non_contig = x.permute(0, 2, 1, 3, 4)
    cuda_module = build_kernel()
    out_cuda = cuda_module.forward(x_non_contig, kernel_size, stride, padding)
    out_reference = cuda_module.forward(x_non_contig.contiguous(),
        kernel_size, stride, padding)
    assert torch.allclose(out_cuda, out_reference, atol=0.01
        ), 'Kernel does not exhibit the expected issue with non-contiguous inputs!'
