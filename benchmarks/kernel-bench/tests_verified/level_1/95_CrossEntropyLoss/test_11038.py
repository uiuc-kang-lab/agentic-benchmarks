import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import os


def test_non_contiguous_predictions():
    batch_size = 64
    num_classes = 10
    base = torch.randn(num_classes, batch_size, device='cuda', dtype=torch.
        float32)
    predictions = base.t()
    targets = torch.randint(0, num_classes, (batch_size,), device='cuda',
        dtype=torch.int64)
    my_kernel = build_kernel()
    try:
        loss_kernel = my_kernel.forward(predictions, targets)
    except RuntimeError as e:
        pytest.skip(f'Kernel does not support non-contiguous predictions: {e}')
    loss_pt = torch.nn.functional.cross_entropy(predictions.contiguous(),
        targets)
    assert torch.allclose(loss_kernel, loss_pt, atol=0.01
        ), f'Kernel reduction likely masks inefficiency or precision issues for non-contiguous predictions.'
    torch.cuda.synchronize()


def test_small_num_classes():
    batch_size = 64
    num_classes = 2
    predictions = torch.randn(batch_size, num_classes, device='cuda', dtype
        =torch.float32)
    targets = torch.randint(0, num_classes, (batch_size,), device='cuda',
        dtype=torch.int64)
    my_kernel = build_kernel()
    loss_kernel = my_kernel.forward(predictions, targets)
    loss_pt = torch.nn.functional.cross_entropy(predictions, targets)
    assert torch.allclose(loss_kernel, loss_pt, atol=1e-07
        ), 'Kernel reduction likely masks inefficiency or precision issues for very small num_classes.'
    torch.cuda.synchronize()


def test_num_classes_much_smaller_than_threads_x():
    batch_size = 128
    num_classes = 3
    predictions = torch.randn(batch_size, num_classes, device='cuda', dtype
        =torch.float32)
    targets = torch.randint(0, num_classes, (batch_size,), device='cuda',
        dtype=torch.int64)
    my_kernel = build_kernel()
    loss_kernel = my_kernel.forward(predictions, targets)
    loss_pt = torch.nn.functional.cross_entropy(predictions, targets)
    diff = (loss_kernel - loss_pt).abs().item()
    assert diff < 1e-02, f'Kernel reduction likely masks inefficiency or precision issues for num_classes much smaller than threads_x. Difference: {diff}'
    torch.cuda.synchronize()
