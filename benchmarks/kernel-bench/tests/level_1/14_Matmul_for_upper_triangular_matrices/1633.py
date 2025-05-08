
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build the CUDA extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# -------------------------------
# Test 1: Data type mismatch (non-float32 tensors)
# This test should trigger the issue of tensor type checking.
def test_input_tensor_type():
    cuda_module = build_kernel()
    N = 128
    A = torch.triu(torch.randn(N, N, dtype=torch.double, device="cuda"))
    B = torch.triu(torch.randn(N, N, dtype=torch.double, device="cuda"))
    with pytest.raises(RuntimeError):
        # The kernel expects float pointers. A failure (or type error) is expected.
        C = cuda_module.forward(A, B)
        torch.cuda.synchronize()

# -------------------------------
# Test 2: Non-contiguous tensors
# Create upper triangular matrices that are non-contiguous by transposing them.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    N = 128
    A_full = torch.triu(torch.randn(N, N, dtype=torch.float32, device="cuda"))
    B_full = torch.triu(torch.randn(N, N, dtype=torch.float32, device="cuda"))
    # Transpose to make non-contiguous tensors (transpose of an upper triangular matrix is lower triangular)
    A = A_full.t()
    B = B_full.t()
    # Although the kernel is only valid for contiguous memory of a full upper triangular matrix,
    # we expect the behavior to be wrong (or even trigger a memory fault) because the pointers are not laid out as expected.
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference using matmul and then selecting the upper triangular part.
    C_ref = torch.triu(torch.matmul(A, B))
    # Expect that results are not close.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel unexpectedly handled non-contiguous inputs correctly!"

# -------------------------------
# Test 3: Non-square matrices
# This test forces the kernel to use matrices that are not square.
def test_non_square_matrix():
    cuda_module = build_kernel()
    # Create non-square matrices. For simplicity, assume A is (M x N) and B is (N x M),
    # but the kernel uses A.size(0) to determine the dimension.
    M, N_dim = 128, 192
    # We still create full MxM matrices from the upper triangle of a larger matrix,
    # so force the mistake by embedding non-square data.
    A_full = torch.triu(torch.randn(M, N_dim, dtype=torch.float32, device="cuda"))
    B_full = torch.triu(torch.randn(N_dim, M, dtype=torch.float32, device="cuda"))
    # The kernel is not designed for non-square matrices; it uses A.size(0) as N.
    # We expect either an error or mismatched results.
    C = cuda_module.forward(A_full, B_full)
    torch.cuda.synchronize()
    # Compute reference multiplication in full and then extract an M x M upper triangular matrix.
    C_ref = torch.triu(torch.matmul(A_full, B_full))
    # The results will likely not match due to incorrect usage of dimensions.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel unexpectedly handled non-square matrices correctly!"

# -------------------------------
# Test 4: Kernel launch error reporting (lack of error checking)
# We trigger an error by using a matrix size that leads to an excessive grid dimension.
def test_kernel_launch_error():
    cuda_module = build_kernel()
    # Choose a huge dimension to potentially create an invalid grid configuration.
    N = 100 * 1024  # extremely large dimension likely to exceed grid limits
    try:
        A = torch.triu(torch.randn(N, N, dtype=torch.float32, device="cuda"))
        B = torch.triu(torch.randn(N, N, dtype=torch.float32, device="cuda"))
    except RuntimeError:
        pytest.skip("Skipping test_kernel_launch_error due to insufficient device memory")

    # Launch the kernel which may silently fail due to invalid grid dimensions since no error checking is done.
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    # Check if the resulting tensor is all zeros (which might be the case if kernel launch silently failed)
    if torch.all(C == 0):
        pytest.fail("Kernel launch error might have occurred and was not reported!")
