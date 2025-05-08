import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def reference_layernorm(x, weight, bias, eps):
    normalized_shape = weight.shape
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias,
        eps)


def test_noncontiguous_input():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    torch.cuda.synchronize()
    batch = 16
    normalized_shape = 8, 16
    x = torch.randn(batch, *normalized_shape, device='cuda')
    x = x.transpose(0, 1)
    weight = torch.randn(*normalized_shape, device='cuda')
    bias = torch.randn(*normalized_shape, device='cuda')
    my_module = build_kernel()
    try:
        output = my_module.forward(x, weight, bias)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    ref = reference_layernorm(x.contiguous(), weight, bias, 1e-05)
    torch.cuda.synchronize()
    diff = (output - ref).abs().max()
    assert diff > 0.001, f'Noncontiguous input did not trigger the expected error. Max diff: {diff}'


def test_mismatched_normalized_shape():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    torch.cuda.synchronize()
    batch = 16
    normalized_shape = 10,
    x = torch.randn(batch, 9, device='cuda')
    weight = torch.randn(*normalized_shape, device='cuda')
    bias = torch.randn(*normalized_shape, device='cuda')
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = my_module.forward(x, weight, bias)


def test_small_normalized_size():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    torch.cuda.synchronize()
    batch = 32
    normalized_shape = 8,
    x = torch.randn(batch, *normalized_shape, device='cuda')
    weight = torch.randn(*normalized_shape, device='cuda')
    bias = torch.randn(*normalized_shape, device='cuda')
    my_module = build_kernel()
    output = my_module.forward(x, weight, bias)
    ref = reference_layernorm(x, weight, bias, 1e-05)
    torch.cuda.synchronize()
    diff = (output - ref).abs().max()
    assert diff > 0.001, f'Kernel with small normalized_size did not trigger the expected error. Max diff: {diff}'
