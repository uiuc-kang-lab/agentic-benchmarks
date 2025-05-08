import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.nn import functional as F
from torch.utils.cpp_extension import load

@pytest.mark.skipif(not torch.cuda.is_available(), reason=
    'CUDA is not available')
def test_non_contiguous_input():
    batch_size = 2
    in_channels = 3
    out_channels = 2
    kernel_size = 3, 3
    stride = 1, 1
    padding = 1, 1
    x = torch.randn(batch_size, in_channels, 16, 16, device='cuda', dtype=
        torch.float32)
    x_non_contig = x.permute(0, 2, 3, 1)
    x_non_contig = x_non_contig.contiguous().transpose(1, 3)
    assert not x_non_contig.is_contiguous(
        ), 'Test setup error: tensor is unexpectedly contiguous.'
    weight = torch.randn(in_channels, out_channels, *kernel_size, device=
        'cuda', dtype=torch.float32)
    cuda_module = build_kernel()
    try:
        output_non_contig = cuda_module.forward(x_non_contig, weight, None,
            list(stride), list(padding))
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    output_contig = cuda_module.forward(x_non_contig.contiguous(), weight,
        None, list(stride), list(padding))
    diff = (output_non_contig - output_contig).abs().max().item()
    assert diff < 0.01, f'Non-contiguous input produced unexpected results, diff = {diff}'
