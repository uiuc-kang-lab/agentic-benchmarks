
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Compile and load the CUDA kernel from kernel.cu
def build_kernel():
    cuda_module = load(
        name="triangular_mm",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1:
# Test with a matrix size that is not a multiple of TILE_SIZE so that the kernel may access out-of-bound memory.
def test_out_of_bound_access():
    # Choose N so that the tile indexing in A can trigger an out-of-bound load.
    # For instance, with TILE_SIZE=16, a relatively small matrix (e.g. 18x18) may trigger the problematic
    # branch when (t*TILE_SIZE + tx) >= N.
    N = 18
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    # Make lower triangular
    A = torch.tril(A)
    B = torch.tril(B)
    
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expecting a runtime error (illegal memory access or similar) because of out-of-bound read.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2:
# Test passing a non-float32 tensor (e.g., float64) to trigger the kernel’s assumption failure.
def test_dtype_mismatch():
    N = 64
    A = torch.randn(N, N, device="cuda", dtype=torch.float64)
    B = torch.randn(N, N, device="cuda", dtype=torch.float64)
    # Make lower triangular
    A = torch.tril(A)
    B = torch.tril(B)

    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # This should trigger an error due to type mismatch in the kernel since it expects float32.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 3:
# Test with non-contiguous inputs. The kernel assumes contiguous memory.
def test_non_contiguous_input():
    N = 128
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    # Create non-contiguous tensors by transposing
    A_nc = A.t()
    B_nc = B.t()

    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel uses data_ptr<float>() assuming contiguous layout.
        C = my_module.forward(A_nc, B_nc)
        torch.cuda.synchronize()
