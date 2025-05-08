import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load

def test_cuda_stream_misuse():
    my_module = build_kernel()
    A = torch.randn(8192, 8192, device='cuda', dtype=torch.float)
    s = 3.0
    default_stream = torch.cuda.current_stream().cuda_stream
    C = my_module.forward(A, s)
    torch.cuda.synchronize()
    C_ref = A * s
    assert torch.allclose(C, C_ref, atol=0.01
        ), 'Kernel output does not match expected result - possible issue with CUDA stream synchronization.'


if __name__ == '__main__':
    pytest.main([__file__])
