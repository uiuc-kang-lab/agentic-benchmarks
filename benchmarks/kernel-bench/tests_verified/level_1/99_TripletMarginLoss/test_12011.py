import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def reference_triplet_loss(anchor, positive, negative, margin):
    d_pos = (anchor - positive).norm(2, dim=1)
    d_neg = (anchor - negative).norm(2, dim=1)
    loss = torch.clamp(d_pos - d_neg + margin, min=0)
    return loss.mean()

def test_reduction_overflow():
    kernel_module = build_kernel()
    batch_size = 4
    feat_size = 256 * 40
    anchor = torch.randn(batch_size, feat_size, device='cuda', dtype=torch.
        float32)
    positive = torch.randn(batch_size, feat_size, device='cuda', dtype=
        torch.float32)
    negative = torch.randn(batch_size, feat_size, device='cuda', dtype=
        torch.float32)
    margin = 1.0
    loss = kernel_module.forward(anchor, positive, negative, margin)
    ref_loss = reference_triplet_loss(anchor, positive, negative, margin)
    assert torch.allclose(loss, ref_loss, atol=0.001
        ), f'Test failed: Kernel loss ({loss.item()}) does not match reference loss ({ref_loss.item()}) for large input size.'
