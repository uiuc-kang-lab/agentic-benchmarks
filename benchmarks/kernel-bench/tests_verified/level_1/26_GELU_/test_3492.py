import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_non_contiguous_input():
    x = torch.randn(64, 128, device='cuda')
    x_t = x.t()
    my_module = build_kernel()
    try:
        out = my_module.forward(x_t)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    out_ref = torch.nn.functional.gelu(x_t.contiguous()).view_as(x_t)
    assert torch.allclose(out, out_ref, atol=0.01), (
        'Kernel output differs from expected output for non-contiguous input.'
        f' Max diff: {(out - out_ref).abs().max().item()}'
    )

def test_excessive_grid_dimension():
    threads = 256
    max_grid = 65535
    blocks = max_grid + 1
    numel = threads * blocks
    x = torch.randn(numel, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    out = my_module.forward(x)
    out_ref = torch.nn.functional.gelu(x)
    assert torch.allclose(out, out_ref, atol=0.01), (
        'Kernel output differs from expected output for excessive grid dimension.'
        f' Max diff: {(out - out_ref).abs().max().item()}'
    )


def test_no_explicit_synchronization():
    x = torch.randn(1024, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    out = my_module.forward(x)
    torch.cuda.synchronize()
    out_ref = torch.nn.functional.gelu(x)
    assert torch.allclose(out, out_ref, atol=0.01), (
        'Kernel output differs from expected output without explicit synchronization.'
        f' Max diff: {(out - out_ref).abs().max().item()}'
    )


if __name__ == '__main__':
    pytest.main([__file__])
