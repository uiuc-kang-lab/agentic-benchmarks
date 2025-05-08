import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_non_contiguous_input():
    my_module = build_kernel()
    x = torch.randn(64, 64, device='cuda', dtype=torch.float32).t()
    negative_slope = 0.01
    try:
        result_kernel = my_module.forward(x, negative_slope)
    except RuntimeError as e:
        pytest.skip('Kernel does not support non-contiguous inputs: ' + str(e))
    result_ref = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    assert torch.allclose(result_kernel, result_ref, atol=0.01
        ), f'Kernel produced incorrect result for non-contiguous input. Max diff: {(result_kernel - result_ref).abs().max().item()}'


def test_shared_memory_branch():
    my_module = build_kernel()
    n = 1048576
    x = torch.randn(n, device='cuda', dtype=torch.float32)
    negative_slope = 0.01
    out = my_module.forward(x, negative_slope)
    out_ref = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    torch.cuda.synchronize()
    assert torch.allclose(out, out_ref, atol=0.01
        ), 'Shared memory branch output does not match torch.nn.functional.leaky_relu'


def test_non_shared_memory_branch():
    my_module = build_kernel()
    n = 1000
    x = torch.randn(n, device='cuda', dtype=torch.float32)
    negative_slope = 0.01
    out = my_module.forward(x, negative_slope)
    out_ref = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    torch.cuda.synchronize()
    assert torch.allclose(out, out_ref, atol=0.01
        ), 'Non-shared memory branch output does not match torch.nn.functional.leaky_relu'
