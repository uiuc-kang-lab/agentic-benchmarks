
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# A helper function to compile and load the CUDA extension.
def build_kernel():
    module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Issue 1: The kernel supports only float32.
def test_input_tensor_type():
    # Create double precision (float64) lower triangular matrices.
    N = 256
    A = torch.randn(N, N, dtype=torch.float64, device="cuda")
    B = torch.randn(N, N, dtype=torch.float64, device="cuda")
    # Force them to be lower triangular.
    A = torch.tril(A)
    B = torch.tril(B)
    module = build_kernel()
    
    # The kernel expects float32. Converting the expected result to float32.
    # Since the kernel will misinterpret the memory layout of double tensors,
    # its output is expected to be incorrect.
    C_kernel = module.forward(A, B)
    C_ref = torch.tril(torch.matmul(A.float(), B.float())).to(torch.float32)
    
    # We expect a large difference because the kernel is not handling float64.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-4), (
        "Kernel unexpectedly handled float64 input; issue (1) may be fixed."
    )

# Issue 2: The kernel assumes square inputs.
def test_non_square_input():
    # Create non-square matrices.
    N = 256
    M = 128  # different from N
    # Even if we force a triangular mask, the kernel uses A.size(0) for N.
    A = torch.randn(N, M, dtype=torch.float32, device="cuda")
    B = torch.randn(N, M, dtype=torch.float32, device="cuda")
    # Apply a lower triangular mask on a rectangular matrix (only along the first two dims).
    A = torch.tril(A)
    B = torch.tril(B)
    module = build_kernel()
    
    with pytest.raises(RuntimeError):
        # Expect an error due to dimension mismatch (since the kernel assumes square shapes)
        _ = module.forward(A, B)

# Issue 3: The kernel is specialized for lower triangular multiplication.
def test_general_full_matrix_multiplication():
    # Create full (non-triangular) matrices.
    N = 256
    A = torch.randn(N, N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, N, dtype=torch.float32, device="cuda")
    module = build_kernel()
    
    # Note: The kernel internally performs a summation from col to row
    # and then explicitly zeros out elements with row < col.
    # Therefore, even for full matrices, its output will be the lower triangular portion
    # of the true matrix product.
    C_kernel = module.forward(A, B)
    C_ref = torch.tril(torch.matmul(A, B))
    
    # The result from the kernel should match the lower triangular version.
    # We expect them to be similar.
    assert torch.allclose(C_kernel, C_ref, atol=1e-4), (
        "Kernel output does not match expected lower triangular multiplication result; "
        "this highlights its non-general design."
    )

# Issue 4: No actual overlapping of memory transfers is performed.
# While it is not straightforward to trigger this issue from Python,
# we can at least check that the kernel runs in a reasonable time for a sizable problem.
def test_no_memory_transfer_overlap():
    N = 2048
    A = torch.randn(N, N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, N, dtype=torch.float32, device="cuda")
    A = torch.tril(A)
    B = torch.tril(B)
    module = build_kernel()
    
    # Measure execution time (this is a loose check; if there were real overlapping,
    # we might expect performance gains that are not present).
    torch.cuda.synchronize()
    import time
    start = time.time()
    C_kernel = module.forward(A, B)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # We set an arbitrary threshold; in an ideal overlapping implementation,
    # the kernel might handle larger sizes faster.
    # If the kernel were truly overlapping memory transfers, one might expect a lower time.
    assert elapsed > 0.01, (
        "Kernel execution is suspiciously fast, possibly indicating an unintended overlap behavior."
    )

# Issue 5: The kernel does not perform error checking on CUDA API calls.
# This issue is internal to the kernel. However, if a CUDA API error occurred,
# the function should throw an error. We simulate an error condition by passing an
# excessively large input size that may force a kernel launch failure.
def test_cuda_error_handling():
    # Try to force a kernel launch error by using a huge size.
    # This size is arbitrarily huge to likely trigger a CUDA resource error.
    N = 100000
    try:
        A = torch.randn(N, N, dtype=torch.float32, device="cuda")
        B = torch.randn(N, N, dtype=torch.float32, device="cuda")
        A = torch.tril(A)
        B = torch.tril(B)
    except RuntimeError:
        pytest.skip("Skipping test_cuda_error_handling due to inability to allocate huge tensors on this device")
    module = build_kernel()
    with pytest.raises(Exception):
        _ = module.forward(A, B)
