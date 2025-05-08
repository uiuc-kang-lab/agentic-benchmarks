import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_incorrect_offset_calculation():
    """
    Trigger Issue 1:
    Use a higher-dimensional tensor where the cumulative product dimension is not the last dimension.
    The kernel’s simplistic offset calculation will produce a wrong result.
    """
    my_module = build_kernel()
    input_tensor = torch.randn(2, 3, 4, device='cuda', dtype=torch.float32)
    expected = torch.cumprod(input_tensor, dim=1)
    output = my_module.forward(input_tensor, 1)
    assert torch.allclose(output, expected, atol=0.01
        ), 'Kernel output does not match expected result for incorrect offset calculation.'

def test_invalid_dimension():
    """
    Trigger Issue 3:
    Pass an invalid dimension index to the kernel.
    The kernel doesn’t validate the dimension argument and may access out-of-range memory.
    """
    my_module = build_kernel()
    input_tensor = torch.randn(5, 10, device='cuda', dtype=torch.float32)
    with pytest.raises(Exception):
        _ = my_module.forward(input_tensor, 2)
