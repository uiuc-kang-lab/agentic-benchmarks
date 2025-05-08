import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_hardsigmoid_clamp_correctness():
    input_tensor = torch.tensor([-10.0, -3.0, 0.0, 3.0, 10.0], device=
        'cuda', dtype=torch.float32)
    expected = ((input_tensor + 3.0) / 6.0).clamp(min=0, max=1)
    module = build_kernel()
    output = module.forward(input_tensor)
    torch.cuda.synchronize()
    assert torch.allclose(output, expected, atol=0.01
        ), f'Output {output} does not match expected value {expected}.'

def test_small_tensor_launch_bounds():
    input_tensor = torch.tensor([0.0], device='cuda', dtype=torch.float32)
    expected = ((input_tensor + 3.0) / 6.0).clamp(min=0, max=1)
    module = build_kernel()
    output = module.forward(input_tensor)
    torch.cuda.synchronize()
    assert torch.allclose(output, expected, atol=0.01
        ), f'Output {output} does not match expected value {expected}.'
