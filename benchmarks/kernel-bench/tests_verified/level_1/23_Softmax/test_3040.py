import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import os
import pytest
import torch
from torch.utils.cpp_extension import load


def test_divergent_syncthreads():
    cuda_module = build_kernel()
    batch_size = 4
    num_features = 10
    x = torch.randn(batch_size, num_features, device='cuda', dtype=torch.
        float32)
    try:
        y = cuda_module.forward(x)
        sums = y.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=0.01
            ), 'Output rows do not sum to 1.'
    except Exception as e:
        pytest.fail(
            f'Kernel execution failed (potential __syncthreads divergence): {e}'
            )

def test_invalid_data_access():
    cuda_module = build_kernel()
    batch_size = 4
    num_features = 128
    x = torch.randn(batch_size, num_features, device='cuda', dtype=torch.
        float32).t().contiguous().t()
    try:
        y = cuda_module.forward(x)
        sums = y.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=0.01
            ), 'Output rows do not sum to 1 for non-contiguous input.'
    except Exception as e:
        pytest.fail(
            f'Kernel execution failed on non-standard memory layouts: {e}')


def test_input_dtype():
    cuda_module = build_kernel()
    batch_size = 4
    num_features = 256
    x = torch.randn(batch_size, num_features, device='cuda', dtype=torch.
        float64)
    try:
        y = cuda_module.forward(x)
        sums = y.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=0.01
            ), 'Output rows do not sum to 1 for double precision input.'
    except RuntimeError as e:
        pytest.skip(
            f'Kernel execution failed on double precision input: {e}'
            )


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
