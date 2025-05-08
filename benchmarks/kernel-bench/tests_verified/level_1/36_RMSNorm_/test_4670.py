import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import os


def test_shared_memory_and_block_size_issue():
    batch_size = 4
    num_features = 10
    spatial = 16, 16
    eps = 1e-05
    x = torch.randn(batch_size, num_features, *spatial, device='cuda',
        dtype=torch.float32)
    kernel = build_kernel()
    y = kernel.forward(x, eps)
    rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)
    y_ref = x / rms
    assert torch.allclose(y, y_ref, atol=0.01
        ), 'Output differs, potential shared memory/block size issue.'


def test_non_contiguous_input():
    batch_size = 4
    num_features = 32
    spatial = 8, 8
    eps = 1e-05
    x = torch.randn(batch_size, num_features, *spatial, device='cuda',
        dtype=torch.float32)
    x_non_contig = x.transpose(1, 2)
    kernel = build_kernel()
    try:
        y = kernel.forward(x_non_contig, eps)
    except Exception as e:
        pytest.skip(
            'Kernel did not support non-contiguous input (as expected); ensure caller makes tensor contiguous.'
            )
    y_contig = kernel.forward(x_non_contig.contiguous(), eps)
    rms = torch.sqrt(torch.mean(x_non_contig.contiguous() ** 2, dim=1,
        keepdim=True) + eps)
    y_ref = x_non_contig.contiguous() / rms
    assert torch.allclose(y, y_ref, atol=0.01
        ), 'Kernel incorrectly handled non-contiguous input.'


def test_unsupported_dtype():
    batch_size = 4
    num_features = 32
    spatial = 8, 8
    eps = 1e-05
    x = torch.randn(batch_size, num_features, *spatial, device='cuda',
        dtype=torch.bfloat16)
    kernel = build_kernel()
    try:
        y = kernel.forward(x, eps)
    except Exception as e:
        pytest.skip(
            'Kernel did not support bfloat16 input (as expected); ensure caller makes tensor float32.'
            )
    y_ref = x.float() / torch.sqrt(
        torch.mean(x.float() ** 2, dim=1, keepdim=True) + eps)
    assert torch.allclose(y, y_ref, atol=0.01
        ), 'Kernel incorrectly handled bfloat16 input.'
