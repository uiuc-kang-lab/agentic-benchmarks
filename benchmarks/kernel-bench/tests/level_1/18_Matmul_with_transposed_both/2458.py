
import torch
import pytest
from torch.utils.cpp_extension import load

# Global parameters (same as in the provided PyTorch code)
M = 1024
K = 4096
N = 2048

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

###########################################################################
# Test 1: Non‐contiguous input tensors (Issue 1)
# The kernel assumes contiguous memory layout. We create non‐contiguous tensors
# that nevertheless have the “logical” shape expected by the kernel.
###########################################################################
def test_non_contiguous_input():
    # To simulate non‐contiguity we embed the data in a 3D tensor and then slice out a view.
    # This way the resulting tensor has the proper shape but its memory layout is not contiguous.
    A_big = torch.randn(K, M, 2, device="cuda", dtype=torch.float32)
    B_big = torch.randn(N, K, 2, device="cuda", dtype=torch.float32)
    # Slicing along the last dimension yields a non‐contiguous view with the expected shape.
    A_nc = A_big[..., 0]  # shape (K, M)
    B_nc = B_big[..., 0]  # shape (N, K)
    # Verify that the views are not contiguous.
    assert not A_nc.is_contiguous(), "A_nc is unexpectedly contiguous!"
    assert not B_nc.is_contiguous(), "B_nc is unexpectedly contiguous!"

    # The kernel expects A (shape: (K,M)) and B (shape: (N,K)) to compute C = A.T * B.T,
    # which is equivalent to C[i,j] = sum_k A[k,i]*B[j,k].
    my_module = build_kernel()
    C_kernel = my_module.forward(A_nc, B_nc)
    torch.cuda.synchronize()

    # Compute the reference result using torch.matmul on the transposed inputs.
    C_ref = torch.matmul(A_nc.t(), B_nc.t())
    # Since the kernel assumes contiguity, using non‐contiguous inputs is likely to produce a wrong result.
    # We check that the outputs differ.
    diff = (C_kernel - C_ref).abs().max().item()
    assert diff > 1e-3, f"Kernel output appears correct with non‐contiguous inputs (max diff {diff}); expected an error."

###########################################################################
# Test 2: Deprecated dispatch using A.type() (Issue 2)
# Using A.type() instead of A.scalar_type() might work for common dtypes,
# but it is not future‐proof. Here we force a “less common” floating type
# (e.g. torch.complex64 is not supported) to trigger a dispatch error.
###########################################################################
def test_deprecated_dispatch():
    # Create inputs with an unsupported (for this kernel) type.
    # For example, using a complex dtype should trigger a dispatch error.
    A = torch.randn(K, M, device="cuda", dtype=torch.complex64)
    B = torch.randn(N, K, device="cuda", dtype=torch.complex64)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel should not be able to dispatch for complex types.
        _ = my_module.forward(A, B)
    # If no error is raised, then the dispatch mechanism is likely masking potential issues.

###########################################################################
# Test 3: Dimension mismatch (Issue 3)
# The kernel assumes that A and B have compatible dimensions.
# We deliberately pass mismatched dimensions to see if the lack of dimension checking
# creates a runtime error or produces a wrong result.
###########################################################################
def test_dimension_mismatch():
    # A is expected to be of shape (K, M) and B of shape (N, K).
    # We provide a B with an incorrect second dimension (K + 1 instead of K).
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B_bad = torch.randn(N, K + 1, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    # The kernel does not check dimensions so it may run and write out‐of‐bounds.
    # We try to catch the error. Depending on the system this might throw a CUDA error.
    with pytest.raises(Exception):
        _ = my_module.forward(A, B_bad)
    # If no error is raised, the kernel is producing an incorrect result silently.

###########################################################################
# Test 4: Lack of kernel error checking (Issue 4)
# If there is an error during kernel execution an appropriate error message is desired.
# Here we force an error by providing extremely small grid dimensions
# that lead to an out‐of‐boundary access (by using tensors with zero dimensions).
###########################################################################
def test_kernel_launch_error():
    # Create empty inputs – this is an edge case that should be caught or at least cause a CUDA error.
    A_empty = torch.empty((0, M), device="cuda", dtype=torch.float32)
    B_empty = torch.empty((N, 0), device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    with pytest.raises(Exception):
        _ = my_module.forward(A_empty, B_empty)
    # This test ensures that if the kernel launch encounters an error,
    # the lack of error checking becomes apparent.

if __name__ == "__main__":
    pytest.main([__file__])
