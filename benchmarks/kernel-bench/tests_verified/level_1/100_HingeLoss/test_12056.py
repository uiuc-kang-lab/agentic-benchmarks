import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def run_forward(kernel_module, predictions, targets):
    result = kernel_module.forward(predictions, targets)
    torch.cuda.synchronize()
    return result

def test_mismatched_shapes():
    kernel_module = build_kernel()
    predictions = torch.randn(129, 1, dtype=torch.float32, device='cuda')
    targets = torch.randint(0, 2, (128,), device='cuda', dtype=torch.float32
        ) * 2 - 1
    result = run_forward(kernel_module, predictions, targets)
    expected = torch.mean(torch.clamp(1 - predictions * targets, min=0))
    assert torch.allclose(result, expected, atol=0.01
        ), 'Kernel output does not match reference for mismatched shapes.'


def test_non_contiguous_inputs():
    kernel_module = build_kernel()
    batch_size = 128
    predictions_full = torch.randn(batch_size * 2, 1, dtype=torch.float32,
        device='cuda')
    targets_full = torch.randint(0, 2, (batch_size * 2,), device='cuda',
        dtype=torch.float32) * 2 - 1
    predictions = predictions_full[::2]
    targets = targets_full[::2]
    try:
        result = run_forward(kernel_module, predictions, targets)
    except RuntimeError as e:
        pytest.skip(f'Kernel does not support non-contiguous inputs: {e}')
    expected = torch.mean(torch.clamp(1 - predictions * targets, min=0))
    assert torch.allclose(result, expected, atol=0.01
        ), 'Kernel output does not match reference for non-contiguous inputs.'

if __name__ == '__main__':
    pytest.main([__file__])
