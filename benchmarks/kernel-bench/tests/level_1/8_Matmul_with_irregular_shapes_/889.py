
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build and load the CUDA extension from kernel.cu
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Unsupported data type (only float32 is supported)
def test_input_tensor_type():
    kernel_module = build_kernel()
    # Create double precision tensors on CUDA. The kernel expects float32.
    M, K, N = 64, 128, 32
    A = torch.randn(M, K, device="cuda", dtype=torch.double)
    B = torch.randn(K, N, device="cuda", dtype=torch.double)

    # Run the kernel. Even if it does not crash, the result will not match the true matmul.
    C = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A.float(), B.float())  # force correct type computation

    # The output C (computed as float*) from double inputs is expected to be wrong.
    # We check that the result is not close.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel should not support float64 but produced close result."

# Issue 2: Non–contiguous (or higher–dimensional) input tensors
def test_noncontiguous_input():
    kernel_module = build_kernel()
    M, K, N = 71, 59, 83  # dimensions not multiples of tile sizes to force boundary conditions

    # Create a contiguous tensor and then take a slice to produce a non–contiguous tensor.
    A_full = torch.randn(M + 2, K + 2, device="cuda", dtype=torch.float32)
    B_full = torch.randn(K + 2, N + 2, device="cuda", dtype=torch.float32)
    # Slicing to get a non–contiguous tensor view of shape (M, K)
    A = A_full[1:M+1, 1:K+1]
    B = B_full[1:K+1, 1:N+1]

    # Even though the kernel does not validate contiguity,
    # using non–contiguous inputs may lead to wrong results.
    C = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)

    # We expect the kernel output to be incorrect if non–contiguous data accesses are not handled.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel should fail for non-contiguous inputs but produced close result."

def test_invalid_dimensions():
    kernel_module = build_kernel()
    # Create a 3D tensor instead of a 2D matrix to trigger unexpected behavior.
    A = torch.randn(10, 20, 30, device="cuda", dtype=torch.float32)
    B = torch.randn(30, 40, device="cuda", dtype=torch.float32)
    with pytest.raises(IndexError):
        # The kernel expects A to be 2D. Accessing size(0) and size(1) will work but the multiplication will be invalid.
        # Alternatively, this might result in a wrong dimension computation.
        C = kernel_module.forward(A, B)
        torch.cuda.synchronize()
        # If no error is thrown, check that the output shape is not as expected.
        if C.dim() == 2:
            raise IndexError("Kernel accepted 3D input unexpectedly.")

# Issue 3: No post–launch error checking
def test_missing_cuda_error_checking(monkeypatch):
    kernel_module = build_kernel()
    # For this test, we simulate a bad launch by passing mismatched dimensions.
    M, K, N = 32, 16, 0  # N=0 is an invalid dimension
    A = torch.randn(M, K, device="cuda", dtype=torch.float32")
    B = torch.randn(K, N, device="cuda", dtype=torch.float32")
    # The kernel likely will simply do nothing or produce an incorrect result,
    # but without error checking, the user will not be alerted.
    C = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    # Check that the output tensor has shape (M, N) = (32, 0). If not, then something went wrong.
    assert C.shape[0] == M and C.shape[1] == N, "Kernel did not handle zero–dimension correctly, indicating missing error checks."

# Issue 4: Fixed tile sizes may cause incorrect results on irregular shapes.
def test_irregular_dimensions():
    kernel_module = build_kernel()
    # Create matrices whose dimensions are not multiples of 32 (tile size) or sub–tile (2)
    M, K, N = 73, 89, 101
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # With the fixed tiling parameters, the boundary loads may be imprecise.
    assert not torch.allclose(C, C_ref, atol=1e-4), "Kernel with fixed tile sizes should produce errors on irregular dimensions."

