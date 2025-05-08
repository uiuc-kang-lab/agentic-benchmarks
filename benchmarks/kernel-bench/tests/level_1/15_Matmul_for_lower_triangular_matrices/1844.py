
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Ensure that non-float32 inputs (e.g., float64) lead to incorrect results.
# This test should trigger the issue of hard-coded float32 assumptions.
def test_non_float32_dtype():
    N = 256
    A = torch.randn(N, N, dtype=torch.float64, device="cuda")
    B = torch.randn(N, N, dtype=torch.float64, device="cuda")
    # Make lower triangular
    A = torch.tril(A)
    B = torch.tril(B)
    my_module = build_kernel()
    # The kernel will reinterpret the double memory as floats, causing significant error.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.tril(torch.matmul(A.float(), B.float()))
    # We expect a significant difference because inputs were double but computed as float.
    diff = (C - C_ref).abs().max().item()
    assert diff > 1e-2, f"Kernel did not trigger issue with non-float32 input, max diff: {diff}"

# Test case 2: Ensure that non-square matrices are rejected.
def test_non_square_input():
    N = 256
    M = 128
    A = torch.randn(N, M, dtype=torch.float32, device="cuda")
    B = torch.randn(N, M, dtype=torch.float32, device="cuda")
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = my_module.forward(A, B)

# Test case 3: Ensure that mismatched input sizes are rejected.
def test_mismatched_input_sizes():
    N = 256
    A = torch.randn(N, N, dtype=torch.float32, device="cuda")
    B = torch.randn(N+1, N+1, dtype=torch.float32, device="cuda")
    A = torch.tril(A)
    B = torch.tril(B)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = my_module.forward(A, B)

# Test case 4: Check behavior for matrices with dimensions that are not multiples of the launch block size.
def test_non_multiple_of_blocksize():
    # Using a size that is not multiple of 16 (the block size used in the kernel launch)
    N = 70
    A = torch.randn(N, N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, N, dtype=torch.float32, device="cuda")
    A = torch.tril(A)
    B = torch.tril(B)
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # For reference, do full matmul and then take lower triangular part.
    C_ref = torch.tril(torch.matmul(A, B))
    assert torch.allclose(C, C_ref, atol=1e-3), f"Kernel output differs from reference output! Max difference: {(C - C_ref).abs().max().item()}"
