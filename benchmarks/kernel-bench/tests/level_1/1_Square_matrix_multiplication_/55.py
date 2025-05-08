
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension. Make sure kernel.cu is in the current directory.
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

def test_non_float32_input():
    # Issue 1: Kernel only supports float32. Passing double precision should trigger an error.
    N = 64
    A = torch.randn(N, N, dtype=torch.float64, device='cuda')
    B = torch.randn(N, N, dtype=torch.float64, device='cuda')
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        mod.forward(A, B)

def test_non_square_input():
    # Issue 2: Kernel only supports square matrices.
    # Here we construct non-square matrices by making the second matrix have a different column dimension.
    # Even though torch.matmul supports rectangular matmul, our kernel assumes square input.
    N = 64
    A = torch.randn(N, N, dtype=torch.float32, device='cuda')
    B = torch.randn(N, N // 2, dtype=torch.float32, device='cuda')
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        mod.forward(A, B)

def test_batched_input():
    # Issue 3: Kernel does not support batched inputs. Providing a 3D tensor should trigger an error.
    batch = 10
    N = 32
    A = torch.randn(batch, N, N, dtype=torch.float32, device='cuda')
    B = torch.randn(batch, N, N, dtype=torch.float32, device='cuda')
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        mod.forward(A, B)

def test_non_contiguous_input():
    # Issue 4: The kernel requires contiguous tensors.
    # Creating a non-contiguous tensor by transposing will trigger an error.
    N = 64
    A = torch.randn(N, N, dtype=torch.float32, device='cuda').t()
    B = torch.randn(N, N, dtype=torch.float32, device='cuda').t()
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        mod.forward(A, B)
