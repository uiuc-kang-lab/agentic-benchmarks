
import pytest
import torch
from torch.utils.cpp_extension import load
import subprocess
import sys
import os

# Utility to build and load the CUDA kernel module
def build_kernel():
    cuda_module = load(
        name="triangular_mm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Early __syncthreads() mismatches due to early returns.
# For an input where each block has a mix of threads that immediately return (e.g. lower triangular condition)
# and threads that enter the compute loop, deadlock is likely.
# We use a small square matrix (e.g. 32x32) so that within the first block many threads have row < col.
@pytest.mark.timeout(5)
def test_deadlock_due_to_early_return():
    N = 32  # exactly one block; threads with threadIdx.y < threadIdx.x will return early.
    # Build lower triangular matrices.
    A = torch.randn(N, N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, N, dtype=torch.float32, device="cuda")
    A = torch.tril(A)
    B = torch.tril(B)

    module = build_kernel()
    # This call should hang or error out because of mismatched __syncthreads() usage.
    with pytest.raises(Exception):
        C = module.forward(A, B)
        torch.cuda.synchronize()  # force synchronization to catch deadlock/hang

# Issue 2: Kernel requires float32 input, but does not support other dtypes.
def test_input_tensor_type():
    N = 64
    A = torch.randn(N, N, dtype=torch.double, device="cuda")
    B = torch.randn(N, N, dtype=torch.double, device="cuda")
    A = torch.tril(A)
    B = torch.tril(B)

    module = build_kernel()
    with pytest.raises(RuntimeError) as exc_info:
        # The TORCH_CHECK in C++ should trigger an error because A and B are not CUDA float tensors.
        C = module.forward(A, B)
        torch.cuda.synchronize()

    assert "float" in str(exc_info.value).lower(), "Expected an error regarding unsupported data type."

# Issue 3: Kernel assumes contiguous input.
def test_non_contiguous_input():
    N = 128
    A = torch.randn(N, N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, N, dtype=torch.float32, device="cuda")
    A = torch.tril(A)
    B = torch.tril(B)

    # Create non-contiguous views by transposing.
    A_noncontig = A.t()  # transpose makes it non-contiguous
    B_noncontig = B.t()

    module = build_kernel()
    # Although the underlying memory buffer is valid, the simple index arithmetic in the kernel (using data_ptr)
    # will be wrong since it assummes contiguous row-major layout.
    C = module.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()

    # Compute reference result using PyTorch's matmul then tril
    C_ref = torch.tril(torch.matmul(A_noncontig, B_noncontig))
    # The result is expected to be incorrect because the kernel does not handle non-contiguous inputs.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel produced correct result for non-contiguous input (unexpected); it should fail or produce incorrect output."
