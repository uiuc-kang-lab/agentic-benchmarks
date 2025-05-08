
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to compile and load the CUDA extension.
def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: Test with non-float32 tensors.
# The kernel assumes float32 but does not check.
# Passing float64 tensors should trigger a wrong result.
def test_non_float32_dtype():
    # Create double tensors on CUDA.
    M, K, N = 64, 32, 64
    A = torch.randn(M, K, dtype=torch.float64, device="cuda")
    B = torch.randn(K, N, dtype=torch.float64, device="cuda")

    kernel_module = build_kernel()

    # Since the kernel will treat the underlying data as float32,
    # reinterpretation of data will lead to an incorrect result.
    C_custom = kernel_module.forward(A, B)
    C_ref = torch.matmul(A.float(), B.float())  # Convert ref to float32 for fair comparison

    # We expect a significant deviation between C_custom and C_ref.
    # The assertion is that the outputs do not match.
    assert not torch.allclose(C_custom, C_ref, atol=1e-3), \
        "Test failed: The kernel returned a result close to the reference output despite non-float32 inputs."

# Issue 2: Test the cuBLAS fallback branch.
# With matrices larger than MATRIX_SIZE_THRESHOLD for M and N,
# the kernel calls cublasSgemm. Because cuBLAS assumes column-major order,
# the result will be incorrect (transposed or otherwise wrong).
def test_cublas_fallback_transpose_issue():
    # Choose dimensions so that M and N exceed the threshold (512).
    M, K, N = 1024, 32, 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    kernel_module = build_kernel()
    C_custom = kernel_module.forward(A, B)
    C_ref = torch.matmul(A, B)

    # The fallback cuBLAS call is expected to be wrong due to row-major vs column-major handling.
    # Hence, the custom result should differ significantly from the reference.
    max_diff = (C_custom - C_ref).abs().max().item()
    assert not torch.allclose(C_custom, C_ref, atol=1e-3), \
        f"Test failed: The cuBLAS fallback returned a result that is too close to the reference! Max difference: {max_diff}"

# Issue 3: (Optional) Testing for CUDA errors is hard to trigger from Python,
# but we can wrap a kernel call and force an error by providing wrongly sized tensors.
def test_incorrect_matrix_dimensions():
    # Create tensors with mismatched inner dimensions.
    A = torch.randn(64, 32, device="cuda", dtype=torch.float32)
    B = torch.randn(64, 64, device="cuda", dtype=torch.float32)  # Wrong inner dimension

    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect the kernel to fail (or produce erroneous memory accesses)
        # Note: This test may not always trigger an immediate error,
        # but it is designed to expose the lack of dimension checking in the kernel.
        kernel_module.forward(A, B)
