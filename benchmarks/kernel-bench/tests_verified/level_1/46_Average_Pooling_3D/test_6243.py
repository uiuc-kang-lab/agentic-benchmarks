import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.nn import AvgPool3d
from torch.utils.cpp_extension import load


@pytest.fixture(scope='module')
def cuda_module():
    mod = build_kernel()
    return mod


@pytest.fixture
def default_params():
    return 3, 2, 1


def test_shared_memory_uninitialized(cuda_module, default_params):
    batch_size = 2
    channels = 3
    depth = 5
    height = 8
    width = 8
    kernel_size, stride, padding = default_params
    input_tensor = torch.randn(batch_size, channels, depth, height, width,
        device='cuda', dtype=torch.float32)
    output_kernel = cuda_module.forward(input_tensor, kernel_size, stride,
        padding)
    avg_pool = AvgPool3d(kernel_size=kernel_size, stride=stride, padding=
        padding)
    output_ref = avg_pool(input_tensor)
    assert torch.allclose(output_kernel, output_ref, atol=0.01
        ), f'Expected mismatch due to uninitialized shared memory usage, but outputs matched.'

