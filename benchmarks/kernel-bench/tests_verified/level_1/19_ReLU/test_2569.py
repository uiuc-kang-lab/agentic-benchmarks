import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import os


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_misaligned_memory():
    module = build_kernel()
    base = torch.randn(1000 + 1, dtype=torch.float32, device='cuda')
    misaligned = base[1:]
    result_kernel = module.forward(misaligned)
    torch.cuda.synchronize()
    result_ref = torch.relu(misaligned)
    assert torch.allclose(result_kernel, result_ref, atol=0.01
        ), f'Misaligned memory test failed: max diff {(result_kernel - result_ref).abs().max().item()}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_non_contiguous_tensor():
    module = build_kernel()
    x = torch.randn(32, 32, dtype=torch.float32, device='cuda')
    non_contig = x.t()
    try:
        result_kernel = module.forward(non_contig)
    except RuntimeError:
        print("RuntimeError: Kernel does not support non-contiguous inputs.")
        return
    torch.cuda.synchronize()
    result_ref = torch.relu(non_contig)
    assert torch.allclose(result_kernel, result_ref, atol=0.01
        ), f'Non-contiguous tensor test failed: max diff {(result_kernel - result_ref).abs().max().item()}'

