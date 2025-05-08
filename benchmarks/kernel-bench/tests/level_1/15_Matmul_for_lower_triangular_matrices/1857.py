
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build/load the kernel module from kernel.cu
def build_kernel():
    # Assume the current folder contains kernel.cu.
    cuda_module = load(
        name="triangular_mm_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def cuda_module():
    module = build_kernel()
    return module

# Issue 1:
# Test that the kernel fails or produces wrong results if provided with double precision
def test_input_tensor_dtype(cuda_module):
    N = 64
    A = torch.randn(N, N, dtype=torch.double, device='cuda')
    B = torch.randn(N, N, dtype=torch.double, device='cuda')
    # Even though the forward() performs type-checks at the Python level in our extension,
    # the kernel itself uses data_ptr<float>() so this mismatch should trigger an error or produce wrong output.
    with pytest.raises(RuntimeError):
        # Expect an error because the underlying kernel expects float
        cuda_module.forward(A, B)
        
# Issue 2:
# Test that the kernel gives an incorrect result when provided with full matrices (non–triangular)
def test_non_triangular_input(cuda_module):
    N = 64
    # Create full random matrices (not forcing a lower triangular structure)
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    C_kernel = cuda_module.forward(A, B)
    # The kernel computes: C[i,j] = sum_{k=j}^{i} A[i,k]*B[k,j] for j <= i (and zero elsewhere)
    # The correct full matmul is:
    C_ref = torch.matmul(A, B)
    # For a triangular multiplication, the expected result if we force lower-triangular
    C_expected = torch.tril(C_ref)
    # The result from the kernel should match C_expected only if the input matrices are lower triangular.
    # When using full matrices, the kernel’s hard-coded lower triangular sum misses contributions.
    # So we assert that it does NOT match the full multiplication.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), (
        "Kernel output unexpectedly matches full matmul on non–triangular inputs."
    )
    # And also, if we force lower triangular on the inputs, then it should match.
    A_tri = torch.tril(A)
    B_tri = torch.tril(B)
    C_kernel_tri = cuda_module.forward(A_tri, B_tri)
    C_ref_tri = torch.tril(torch.matmul(A_tri, B_tri))
    assert torch.allclose(C_kernel_tri, C_ref_tri, atol=1e-5), (
        "Kernel output does not match lower triangular multiplication on triangular inputs."
    )

# Issue 3:
# Test that the kernel launch configuration is not flexible: Small matrices (N < 256) can expose boundary issues.
def test_small_matrix(cuda_module):
    N = 17  # Much smaller than 256
    A_tri = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B_tri = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    C_kernel = cuda_module.forward(A_tri, B_tri)
    # Compute expected output:
    C_ref = torch.tril(torch.matmul(A_tri, B_tri))
    assert torch.allclose(C_kernel, C_ref, atol=1e-5), (
        "Kernel output on a small matrix does not match the expected output."
    )

# Issue 4:
# Test the effect of #pragma unroll 8 by using a matrix with rows that produce a reduced inner loop iteration count.
def test_varying_inner_loop_iterations(cuda_module):
    # Here, we choose a small matrix size so that for many rows the inner loop (k from col to row) iterates only a few times.
    N = 8
    A_tri = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B_tri = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    C_kernel = cuda_module.forward(A_tri, B_tri)
    C_ref = torch.tril(torch.matmul(A_tri, B_tri))
    assert torch.allclose(C_kernel, C_ref, atol=1e-5), (
        "Kernel output with low iteration counts in the inner loop does not match expected output."
    )
