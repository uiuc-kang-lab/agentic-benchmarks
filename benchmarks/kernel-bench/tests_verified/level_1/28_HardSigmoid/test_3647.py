import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load

def test_hardsigmoid_correctness():
    my_module = build_kernel()
    x = torch.randn(2048, device='cuda', dtype=torch.float32)
    output = my_module.forward(x)
    expected = torch.nn.functional.hardsigmoid(x)
    assert torch.allclose(output, expected, atol=0.01
        ), 'Kernel output does not match expected HardSigmoid output.'
