import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load

def test_misaligned_memory():
    batch_size = 4
    normalized_size = 64
    outer_size = batch_size
    padded = torch.randn(outer_size, normalized_size + 1, dtype=torch.
        float32, device='cuda')
    x = padded[:, 1:]
    weight = torch.randn(normalized_size, dtype=torch.float32, device='cuda')
    bias = torch.randn(normalized_size, dtype=torch.float32, device='cuda')
    my_kernel = build_kernel()
    out_custom = my_kernel.forward(x, weight, bias, 1e-05)
    ln = torch.nn.LayerNorm(normalized_size, eps=1e-05).to(dtype=torch.
        float32, device='cuda')
    with torch.no_grad():
        ln.weight.copy_(weight)
        ln.bias.copy_(bias)
    out_ref = ln(x)
    assert torch.allclose(out_custom, out_ref, atol=0.01
        ), 'Kernel unexpectedly handled misaligned memory correctly.'


def test_noncontiguous_input():
    batch_size = 4
    normalized_size = 64
    outer_size = batch_size
    x = torch.randn(outer_size, normalized_size, dtype=torch.float32,
        device='cuda')
    x_noncontiguous = x.t()
    x_noncontiguous = x_noncontiguous.t()
    assert not x_noncontiguous.is_contiguous(
        ), 'Tensor is unexpectedly contiguous.'
    weight = torch.randn(normalized_size, dtype=torch.float32, device='cuda')
    bias = torch.randn(normalized_size, dtype=torch.float32, device='cuda')
    my_kernel = build_kernel()
    try:
        out_custom = my_kernel.forward(x_noncontiguous, weight, bias, 1e-05)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    ln = torch.nn.LayerNorm(normalized_size, eps=1e-05).to(dtype=torch.
        float32, device='cuda')
    with torch.no_grad():
        ln.weight.copy_(weight)
        ln.bias.copy_(bias)
    out_ref = ln(x_noncontiguous)
    assert torch.allclose(out_custom, out_ref, atol=0.01
        ), 'Kernel unexpectedly produced correct result for non-contiguous input.'


if __name__ == '__main__':
    pytest.main([__file__])
