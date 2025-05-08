
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension
def build_kernel():
    # Ensure the source file exists
    source_file = os.path.join(os.path.dirname(__file__), "kernel.cu")
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"{source_file} not found.")

    cuda_module = load(
        name="diag_matmul_cuda",
        sources=[source_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that passing non-float32 tensors triggers a problem.
def test_invalid_dtype():
    my_module = build_kernel()
    N = 128
    M = 64
    # Create double precision tensors.
    A = torch.randn(N, dtype=torch.double, device="cuda")
    B = torch.randn(N, M, dtype=torch.double, device="cuda")
    # Expect that calling the kernel with double type will cause an error,
    # because the kernel is expecting float32.
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2: Test that very large tensor dimensions (where N*M overflows 32-bit indexing) lead to incorrect results.
# Note: Allocating a tensor with > 2^31 elements is impractical;
# we simulate the issue by using moderately large dimensions if possible.
@pytest.mark.skipif(True, reason="This test is conceptual since allocating such a tensor may be infeasible. It documents the issue.")
def test_large_dimension_overflow():
    my_module = build_kernel()
    # Choose dimensions where the product would exceed 2**31 if possible.
    # For example, N = 70000, M = 70000 leads to nearly 4.9e9 elements.
    N = 70000
    M = 70000
    A = torch.randn(N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, M, dtype=torch.float32, device="cuda")
    # The kernel uses a 32-bit index for total number of elements.
    # We expect the results to be wrong because of index overflow.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference result using broadcasting multiplication.
    # (note: torch.diag(A) @ B is equivalent to A.unsqueeze(1) * B)
    C_ref = A.unsqueeze(1) * B
    # The maximum absolute difference should be huge because of overflow,
    # so we assert that the kernel result is not close.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel output unexpectedly matches reference for large dimensions!"

# Issue 3: Test non-multiple-of-32 dimensions.
def test_non_multiple_of_32():
    my_module = build_kernel()
    # Choose dimensions where (N*M) is not divisible by 32.
    N = 37  # arbitrary non-multiple of 32 for the number of rows
    M = 53  # arbitrary non-multiple of 32 for number of columns
    A = torch.randn(N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, M, dtype=torch.float32, device="cuda")
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # The expected result: each row i of B is multiplied by A[i]
    C_ref = A.unsqueeze(1) * B
    assert torch.allclose(C, C_ref, atol=1e-5), f"Kernel output differs from reference output! Max diff: {(C-C_ref).abs().max()}"

# Issue 4: Test that the module compiles correctly and in environments where std::min is required.
def test_compilation_min_qualification():
    try:
        my_module = build_kernel()
    except Exception as e:
        pytest.fail(f"Compilation of the CUDA kernel failed possibly due to unqualified 'min': {e}")
