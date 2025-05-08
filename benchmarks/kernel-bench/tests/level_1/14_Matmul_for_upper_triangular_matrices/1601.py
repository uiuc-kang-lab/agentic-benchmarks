
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build and load our CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def cuda_module():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return build_kernel()

# Test 1: Trigger issue with input tensor data type (double instead of float32)
def test_input_tensor_dtype(cuda_module):
    N = 128
    # Create double-precision tensors (non-float32) and force them onto CUDA.
    A = torch.triu(torch.randn(N, N, dtype=torch.float64, device="cuda"))
    B = torch.triu(torch.randn(N, N, dtype=torch.float64, device="cuda"))
    with pytest.raises(RuntimeError):
        # Likely the kernel launch or the pointer conversion will fail.
        C = cuda_module.forward(A, B)
        torch.cuda.synchronize()

# Test 2: Trigger issue with non-contiguous input tensors
def test_non_contiguous_input(cuda_module):
    N = 128
    # Create a contiguous tensor then make a transpose that is non-contiguous.
    A_full = torch.randn(N, N, dtype=torch.float32, device="cuda")
    B_full = torch.randn(N, N, dtype=torch.float32, device="cuda")
    A = torch.triu(A_full).t()  # Transpose makes it non-contiguous.
    B = torch.triu(B_full).t()
    # Even if the values are triangular in the original data,
    # the non-contiguous memory layout may lead to wrong results.
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    # Compare against a reference computed by dense matmul + triu.
    C_ref = torch.triu(torch.matmul(A, B))
    # The results are expected to differ if non-contiguity is not handled correctly.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel should fail for non-contiguous inputs but produced matching result"

# Test 3: Trigger issue with non-square input matrices.
def test_non_square_input(cuda_module):
    M, N = 128, 256  # Non-square dimensions.
    # Create non-square upper-triangular like matrices.
    A = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    B = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    # Fill the upper triangular region in each row.
    for i in range(M):
        if i < N:
            A[i, i:] = torch.randn(N-i)
        if i < N:
            B[i, i:] = torch.randn(N-i)
    with pytest.raises(Exception):
        # The kernel expects a square matrix, so indexing with A.size(0) will be invalid.
        C = cuda_module.forward(A, B)
        torch.cuda.synchronize()

# Test 4: Trigger issue due to lack of CUDA error checking.
def test_cuda_launch_error_detection(cuda_module):
    # We can intentionally force an error by setting N=0.
    N = 0
    A = torch.empty((N, N), dtype=torch.float32, device="cuda")
    B = torch.empty((N, N), dtype=torch.float32, device="cuda")
    # Since no work is scheduled, some kernels may silently pass, but we expect to catch
    # an error or at least a discrepancy in output.
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    assert C.numel() == 0, "Kernel should produce an empty tensor for empty inputs"

# Test 5: Check that over-provisioned threads do not corrupt valid outputs.
def test_overprovisioned_threads(cuda_module):
    # This test will not fail if results are mathematically correct,
    # but will help detect mis-computation caused by stray thread operations.
    N = 128
    A = torch.triu(torch.randn(N, N, dtype=torch.float32, device="cuda"))
    B = torch.triu(torch.randn(N, N, dtype=torch.float32, device="cuda"))
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference result using dense matmul and taking the upper triangular part.
    C_ref = torch.triu(torch.matmul(A, B))
    assert torch.allclose(C, C_ref, atol=1e-4), "Kernel output differs from reference output!"
