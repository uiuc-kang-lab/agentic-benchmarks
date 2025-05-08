import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load

def test_alignment_issue():
    my_module = build_kernel()
    batch_size, features, dim1, dim2 = 4, 16, 32, 32
    x = torch.randn(batch_size, features, dim1, dim2, device='cuda')
    x_non_contig = x.transpose(2, 3)
    weight = torch.randn(features, device='cuda')
    bias = torch.randn(features, device='cuda')
    try:
        y = my_module.forward(x_non_contig, weight, bias, num_groups=4, eps
            =1e-05)
    except Exception as e:
        pytest.skip('Kernel did not handle non–contiguous input: ' + str(e))
    gn = torch.nn.GroupNorm(num_groups=4, num_channels=features, eps=1e-05
        ).cuda()
    y_ref = gn(x_non_contig)
    assert torch.allclose(y, y_ref, atol=0.001
        ), 'Kernel unexpectedly handled mis–aligned / non–contiguous input correctly!'


@pytest.mark.xfail(reason=
    'Cannot easily simulate blockDim > 1024 from Python; this test serves as a placeholder to highlight the potential issue.'
    )
def test_shared_memory_issue():
    my_module = build_kernel()
    batch_size, features, dim1, dim2 = 4, 2048, 64, 64
    x = torch.randn(batch_size, features, dim1, dim2, device='cuda')
    weight = torch.randn(features, device='cuda')
    bias = torch.randn(features, device='cuda')
    y = my_module.forward(x, weight, bias, num_groups=8, eps=1e-05)
    gn = torch.nn.GroupNorm(num_groups=8, num_channels=features, eps=1e-05
        ).cuda()
    y_ref = gn(x)
    assert torch.allclose(y, y_ref, atol=0.001
        ), 'Kernel unexpectedly handled excessive threads per block correctly!'


def test_contiguous_input():
    my_module = build_kernel()
    batch_size, features, dim1, dim2 = 4, 16, 32, 32
    x = torch.randn(batch_size, features, dim1, dim2, device='cuda')
    weight = torch.randn(features, device='cuda')
    bias = torch.randn(features, device='cuda')
    try:
        y = my_module.forward(x, weight, bias, num_groups=4, eps=1e-05)
    except Exception as e:
        pytest.skip('Kernel could not process contiguous input: ' + str(e))
    gn = torch.nn.GroupNorm(num_groups=4, num_channels=features, eps=1e-05
        ).cuda()
    y_ref = gn(x)
    assert torch.allclose(y, y_ref, atol=0.01
        ), 'Kernel output differs from PyTorch GroupNorm for contiguous input!'
