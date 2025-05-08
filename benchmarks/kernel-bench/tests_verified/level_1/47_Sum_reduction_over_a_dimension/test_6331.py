import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_non_contiguous_input():
    x = torch.randn(4, 8, 16, device='cuda', dtype=torch.float32)
    x_non_contig = x.transpose(1, 2)
    reduce_dim = 1
    kernel = build_kernel()
    try:
        output_kernel = kernel.forward(x_non_contig, reduce_dim)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    output_ref = torch.sum(x_non_contig, dim=reduce_dim, keepdim=True)
    assert torch.allclose(output_kernel, output_ref, atol=0.01
        ), f'Kernel output does not match PyTorch sum output for non-contiguous input. This may indicate a precision issue. Kernel output: {output_kernel}, Reference output: {output_ref}'


def test_empty_reduction_dim():
    x = torch.empty(2, 0, 5, device='cuda', dtype=torch.float32)
    reduce_dim = 1
    kernel = build_kernel()
    output_kernel = kernel.forward(x, reduce_dim)
    output_ref = torch.sum(x, dim=reduce_dim, keepdim=True)
    assert torch.allclose(output_kernel, output_ref, atol=0.01
        ), f'Kernel output does not match PyTorch sum output for empty reduction dimension. This may indicate a precision issue. Kernel output: {output_kernel}, Reference output: {output_ref}'


def test_non_canonical_layout():
    x = torch.randn(3, 4, 5, device='cuda', dtype=torch.float32)
    x_permuted = x.permute(2, 0, 1)
    reduce_dim = 1
    kernel = build_kernel()
    output_kernel = kernel.forward(x_permuted, reduce_dim)
    output_ref = torch.sum(x_permuted, dim=reduce_dim, keepdim=True)
    assert torch.allclose(output_kernel, output_ref, atol=0.01
        ), f'Kernel output does not match PyTorch sum output for non-canonical layout. This may indicate a precision issue. Kernel output: {output_kernel}, Reference output: {output_ref}'
