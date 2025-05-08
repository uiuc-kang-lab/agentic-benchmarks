import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load


def test_extreme_values():
    batch_size, dim = 4, 1024
    a = torch.full((batch_size, dim), -1e+30, device='cuda', dtype=torch.
        float32)
    for i in range(batch_size):
        a[i, i] = 1e+30
    my_module = build_kernel()
    out = my_module.forward(a, 1)
    expected = torch.log_softmax(a, dim=1)
    torch.cuda.synchronize()
    assert torch.allclose(out, expected, atol=0.01
        ), 'Extreme values: kernel output does not match expected torch.log_softmax.'


def test_negative_infinities():
    batch_size, dim = 3, 512
    a = torch.full((batch_size, dim), -1e+20, device='cuda', dtype=torch.
        float32)
    for i in range(batch_size):
        a[i, -1] = 0.0
    my_module = build_kernel()
    out = my_module.forward(a, 1)
    expected = torch.log_softmax(a, dim=1)
    torch.cuda.synchronize()
    assert torch.allclose(out, expected, atol=0.01
        ), 'Negative infinity: kernel output does not handle infinities as expected.'


def test_nan_input():
    batch_size, dim = 2, 256
    a = torch.randn(batch_size, dim, device='cuda', dtype=torch.float32)
    a[0, 10] = float('nan')
    a[1, 20] = float('nan')
    my_module = build_kernel()
    out = my_module.forward(a, 1)
    expected = torch.log_softmax(a, dim=1)
    torch.cuda.synchronize()
    assert torch.isnan(out[0, 0]) and torch.isnan(out[1, 0]
        ), 'NaN input: kernel did not produce NaNs as expected in the output.'

if __name__ == '__main__':
    pytest.main([__file__])
