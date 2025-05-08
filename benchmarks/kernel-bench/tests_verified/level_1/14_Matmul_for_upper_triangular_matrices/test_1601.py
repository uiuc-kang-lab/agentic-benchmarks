import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


@pytest.fixture(scope='module')
def cuda_module():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    return build_kernel()

def test_cuda_launch_error_detection(cuda_module):
    N = 0
    A = torch.empty((N, N), dtype=torch.float32, device='cuda')
    B = torch.empty((N, N), dtype=torch.float32, device='cuda')
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    assert C.numel(
        ) == 0, 'Kernel should produce an empty tensor for empty inputs'


def test_overprovisioned_threads(cuda_module):
    N = 128
    A = torch.triu(torch.randn(N, N, dtype=torch.float32, device='cuda'))
    B = torch.triu(torch.randn(N, N, dtype=torch.float32, device='cuda'))
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.triu(torch.matmul(A, B))
    assert torch.allclose(C, C_ref, atol=0.01
        ), f'Kernel output differs from reference output! Max diff: {(C - C_ref).abs().max()}'
