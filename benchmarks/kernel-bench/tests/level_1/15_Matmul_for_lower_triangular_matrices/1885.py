
import torch
import pytest
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Rebuild the CUDA extension each time to catch changes,
    # assuming that kernel.cu is in the current directory.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel is specialized to lower-triangular multiplication.
# If we supply full (non-lower-triangular) matrices, the kernel still zeros out the upper-triangular part.
# Thus, comparing the kernel output with full matrix multiplication (torch.matmul) should fail.
def test_kernel_with_full_matrix_inputs():
    # Use a moderate size to illustrate the problem.
    N = 128
    # Create full matrices (not lower triangular)
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    kernel_module = build_kernel()
    C_kernel = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    # Expected: full matrix multiplication (with full accumulation)
    C_ref = torch.matmul(A, B)
    # The kernel only produces the lower-triangular part of the product
    # so the upper triangular part of C_kernel will be zero.
    # Thus, values in C_kernel in the upper triangle will differ from C_ref.
    diff = (C_kernel - C_ref).abs().max()
    assert diff > 1e-3, f"Test did not trigger Issue 1: the kernel appears to support full matrix inputs."

# Issue 2: Unused constant memory parameters.
# The kernel transfers d_num_chunks and d_chunk_sizes into constant memory,
# but never makes use of them. In a more general setting where tiling is required,
# one expects these parameters to affect computation.
# We trigger this by providing a matrix size that is not a multiple of BLOCK_SIZE,
# where the unused chunk sizes would have been used.
def test_kernel_with_dimension_not_multiple_of_block_size():
    # Choose a size that is not a multiple of BLOCK_SIZE (32)
    N = 45  # Intentionally not a multiple of 32.
    # Create lower-triangular matrices as expected by the kernel.
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    kernel_module = build_kernel()
    C_kernel = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute the expected result exactly as in the Python code.
    C_ref = torch.tril(torch.matmul(A, B))
    # Due to the fixed iteration strategy and ignoring provided chunk size info,
    # the kernel may produce an incorrect result for matrix dimensions not divisible by BLOCK_SIZE.
    # We check that the kernel result does NOT match the expected result.
    diff = (C_kernel - C_ref).abs().max()
    assert diff > 1e-3, f"Test did not trigger Issue 2: the kernel produced nearly correct results even when dimensions are not a multiple of BLOCK_SIZE."

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
