import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load
import os


@pytest.fixture(scope='module')
def kernel_module():
    return build_kernel()


def compute_triplet_loss(anchor, positive, negative, margin):
    """
    Compute the loss per sample using Euclidean distances and the triplet loss formula:
       loss = max(0, ||anchor - positive|| - ||anchor - negative|| + margin)
    Returns: a single scalar that is the mean over the batch.
    """
    return torch.nn.TripletMarginLoss(margin=margin)(anchor=anchor, positive=positive, negative=negative)


def test_shared_memory_reduction_issue(kernel_module):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    batch_size = 8
    feat_size = 64
    margin = 0.5
    anchor = torch.ones(batch_size, feat_size, dtype=torch.float32, device=
        'cuda')
    positive = torch.ones(batch_size, feat_size, dtype=torch.float32,
        device='cuda')
    negative = torch.zeros(batch_size, feat_size, dtype=torch.float32,
        device='cuda')
    expected_loss = compute_triplet_loss(anchor, positive, negative, margin)
    kernel_loss = kernel_module.forward(anchor, positive, negative, margin)
    torch.cuda.synchronize()
    assert torch.allclose(kernel_loss, expected_loss, atol=0.01
        ), f'Kernel loss ({kernel_loss.item()}) unexpectedly matches the correct computation ({expected_loss.item()}); shared memory reduction bug may be hidden.'


def test_block_single_sample(kernel_module):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    batch_size = 1
    feat_size = 512
    margin = 1.0
    anchor = torch.randn(batch_size, feat_size, device='cuda', dtype=torch.
        float32)
    positive = torch.randn(batch_size, feat_size, device='cuda', dtype=
        torch.float32)
    negative = torch.randn(batch_size, feat_size, device='cuda', dtype=
        torch.float32)
    expected_loss = compute_triplet_loss(anchor, positive, negative, margin)
    kernel_loss = kernel_module.forward(anchor, positive, negative, margin)
    torch.cuda.synchronize()
    assert torch.allclose(kernel_loss, expected_loss, atol=0.01
        ), f'Kernel loss ({kernel_loss.item()}) does not match expected ({expected_loss.item()}) in single-sample block scenario.'
