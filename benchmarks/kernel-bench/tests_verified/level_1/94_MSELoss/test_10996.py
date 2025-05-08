import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load
import math


def test_inflexible_block_size():
    N = 10
    preds = torch.linspace(0, 1, steps=N, device='cuda', dtype=torch.float32)
    tgts = torch.linspace(0, 1, steps=N, device='cuda', dtype=torch.float32)
    expected = torch.tensor(0.0, device='cuda', dtype=torch.float32)
    kernel = build_kernel()
    out = kernel.forward(preds, tgts)
    torch.cuda.synchronize()
    assert math.isclose(out.item(), expected.item(), rel_tol=1e-05
        ), f'Issue 1 triggered: Expected {expected.item()}, got {out.item()}'


def test_hard_coded_warp_size():
    N = 256 - 3
    preds = torch.full((N,), 2.0, device='cuda', dtype=torch.float32)
    tgts = torch.full((N,), 1.0, device='cuda', dtype=torch.float32)
    expected = torch.tensor(1.0, device='cuda', dtype=torch.float32)
    kernel = build_kernel()
    out = kernel.forward(preds, tgts)
    torch.cuda.synchronize()
    assert torch.allclose(out, expected, atol=0.01
        ), f'Issue 3 triggered: Expected a mismatch due to warp-size assumptions, got {out} vs {expected}'
