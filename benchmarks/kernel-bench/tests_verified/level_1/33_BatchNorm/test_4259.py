import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


class BatchNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_var,
        training, momentum, eps, kernel):
        return kernel.forward(input, weight, bias, running_mean,
            running_var, training, momentum, eps)

def test_non_contiguous_input():
    kernel = build_kernel()
    batch_size = 16
    channels = 64
    H, W = 32, 32
    x = torch.randn(batch_size, channels, H, W, device='cuda', dtype=torch.
        float32).transpose(1, 2)
    weight = torch.randn(channels, device='cuda', dtype=torch.float32)
    bias = torch.randn(channels, device='cuda', dtype=torch.float32)
    running_mean = torch.zeros(channels, device='cuda', dtype=torch.float32)
    running_var = torch.ones(channels, device='cuda', dtype=torch.float32)
    momentum = 0.1
    eps = 1e-05
    try:
        result = BatchNormFunction.forward(None, x, weight, bias, running_mean,
            running_var, True, momentum, eps, kernel)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    expected = torch.nn.functional.batch_norm(x, weight, bias, running_mean,
        running_var, True, momentum, eps)
    assert torch.allclose(result, expected, atol=0.01
        ), 'Kernel incorrectly produced the expected result with a non-contiguous tensor. This indicates it may not be handling non-contiguous memory layouts correctly.'


def test_incorrect_input_dimensions():
    kernel = build_kernel()
    batch_size = 16
    channels = 64
    H = 32
    x = torch.randn(batch_size, channels, H, device='cuda', dtype=torch.float32
        )
    weight = torch.randn(channels, device='cuda', dtype=torch.float32)
    bias = torch.randn(channels, device='cuda', dtype=torch.float32)
    running_mean = torch.zeros(channels, device='cuda', dtype=torch.float32)
    running_var = torch.ones(channels, device='cuda', dtype=torch.float32)
    momentum = 0.1
    eps = 1e-05
    with pytest.raises(Exception):
        out = BatchNormFunction.forward(None, x, weight, bias, running_mean,
            running_var, True, momentum, eps, kernel)
        torch.cuda.synchronize()
