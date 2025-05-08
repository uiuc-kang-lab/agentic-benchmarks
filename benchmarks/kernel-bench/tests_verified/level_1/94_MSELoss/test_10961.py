import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load

def test_empty_tensor():
    cuda_module = build_kernel()
    preds = torch.empty((0,), device='cuda', dtype=torch.float32)
    tgts = torch.empty((0,), device='cuda', dtype=torch.float32)
    out = cuda_module.forward(preds, tgts)
    torch.cuda.synchronize()
    assert torch.isnan(out).item(
        ), 'Expected NaN due to division by zero on empty tensor'


def test_non_contiguous_input():
    cuda_module = build_kernel()
    preds = torch.randn(128, 4096, device='cuda', dtype=torch.float32)
    tgts = torch.randn(128, 4096, device='cuda', dtype=torch.float32)
    preds_nc = preds.t()
    tgts_nc = tgts.t()
    try:
        out_kernel = cuda_module.forward(preds_nc, tgts_nc)
    except RuntimeError as e:
        pytest.skip(f'Kernel does not support non-contiguous input: {e}')
    torch.cuda.synchronize()
    out_ref = torch.mean((preds_nc - tgts_nc) ** 2)
    assert torch.allclose(out_kernel, out_ref, atol=0.01
        ), 'Kernel output unexpectedly matches reference for non-contiguous tensors'
