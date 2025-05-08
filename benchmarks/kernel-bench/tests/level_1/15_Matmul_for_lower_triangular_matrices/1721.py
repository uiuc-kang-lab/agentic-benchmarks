
import torch
import pytest
from torch.utils.cpp_extension import load

# Build the CUDA extension module from kernel.cu
def build_kernel():
    return load(
        name="triangular_mm_kernel_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

# Test case 1: Matrix larger than 32 x 32 to trigger incomplete computation.
def test_incomplete_computation_large_matrix():
    # Create a lower triangular matrix of size >32
    N = 64  # greater than 32 to trigger the issue
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)

    kernel_module = build_kernel()
    C_kernel = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    
    # Reference: using torch.matmul then extracting lower triangle
    C_ref = torch.tril(torch.matmul(A, B))
    
    # The kernel is expected to miss computing entries for col >= 32 for rows > 32.
    # Therefore, the test should fail if these entries are not computed correctly.
    # We check that the kernel result differs from the reference.
    if torch.allclose(C_kernel, C_ref, atol=1e-5):
        pytest.fail("Kernel incorrectly computed full result for matrix size >32. Expected incomplete computation.")

# Test case 2: Using a non float32 type (e.g., double) to trigger type incompatibility.
def test_incorrect_dtype():
    N = 32  # Even if the size is within 32, using double should cause an error or mismatch.
    A = torch.randn(N, N, device="cuda", dtype=torch.float64)
    B = torch.randn(N, N, device="cuda", dtype=torch.float64)
    A = torch.tril(A)
    B = torch.tril(B)
    
    kernel_module = build_kernel()
    
    with pytest.raises(RuntimeError):
        # The forward function expects float tensors.
        _ = kernel_module.forward(A, B)

# Test case 3: When the number of columns in a row exceeds the available threads.
def test_insufficient_thread_coverage():
    # In this test, we set up a scenario where a specific row has more valid columns
    # than the fixed 32 threads. For example, row 50 in a matrix of size 80 has 51 valid columns,
    # but only threads 0-31 are available.
    N = 80  # greater than 32 to trigger the issue (row 50 specifically)
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    
    kernel_module = build_kernel()
    C_kernel = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    
    C_ref = torch.tril(torch.matmul(A, B))
    
    # Specifically check row 50 for columns beyond thread index 31.
    row_to_check = 50
    # Since threads beyond index 31 are not executed, those entries in kernel output should be zero,
    # while the reference may have nonzero values.
    computed = C_kernel[row_to_check, 32:].cpu()
    reference = C_ref[row_to_check, 32:].cpu()
    
    if torch.allclose(computed, reference, atol=1e-5):
        pytest.fail("Kernel computed the correct values for columns beyond available threads, "
                    "which is unexpected given the fixed block dimension.")
