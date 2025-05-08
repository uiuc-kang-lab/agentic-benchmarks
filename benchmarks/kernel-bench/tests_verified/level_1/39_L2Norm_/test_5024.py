import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load

def test_shared_memory_reduction_issue():
    C = 1024
    input_tensor = torch.ones(1, C, device='cuda', dtype=torch.float32)
    module = build_kernel()
    output = module.forward(input_tensor)
    expected_value = 1.0 / C ** 0.5
    max_diff = (output - expected_value).abs().max().item()
    assert max_diff < 0.01, f'Kernel output differs from expected: max diff {max_diff}. Expected value: {expected_value}'


def test_missing_error_check_issue():
    C = 256
    input_tensor = torch.randn(1, C, device='cuda', dtype=torch.float32)
    module = build_kernel()
    output = module.forward(input_tensor)
    ref_norm = input_tensor.norm(p=2, dim=1, keepdim=True) + 1e-12
    reference = input_tensor / ref_norm
    assert torch.allclose(output, reference, atol=0.01
        ), f'Kernel output differs from reference: max diff {torch.abs(output - reference).max().item()}.'
