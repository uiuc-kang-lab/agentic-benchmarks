
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Utility to compile and load the CUDA extension.
def build_kernel():
    # Rebuild if the .so file exists
    sources = ["kernel.cu"]
    # The extra_cuda_cflags can be adjusted as needed.
    module = load(
        name="triangular_mm",
        sources=sources,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: The upper triangular portion of C is left uninitialized.
def test_uninitialized_upper_triangular():
    # Use a modest matrix size to force a single chunk.
    N = 256
    # Create lower triangular matrices.
    A = torch.randn(N, N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, N, device='cuda', dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    # Expected: full multiplication then take lower triangular.
    expected = torch.tril(torch.matmul(A, B))
    
    kernel_module = build_kernel()
    # Run our custom kernel.
    C = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    
    # Check lower triangular part matches.
    lower_mask = torch.tril(torch.ones_like(C, dtype=torch.bool))
    assert torch.allclose(C[lower_mask], expected[lower_mask], atol=1e-4), \
        "Lower triangular part of output does not match expected result."
    
    # Now check that the upper triangular part is properly set to zero.
    # Since the kernel does not write the upper triangle, these values will be uninitialized.
    upper_mask = ~lower_mask
    # We expect the upper triangle to be zero, but they likely won't be.
    if torch.any(C[upper_mask] != 0):
        pytest.fail("Kernel left the upper triangular part uninitialized (non–zero values found).")
    

# Issue 2: The kernel only supports float32.
def test_single_precision_only():
    N = 128
    # Create double precision lower triangular matrices.
    A = torch.randn(N, N, device='cuda', dtype=torch.float64)
    B = torch.randn(N, N, device='cuda', dtype=torch.float64)
    A = torch.tril(A)
    B = torch.tril(B)
    
    kernel_module = build_kernel()
    
    with pytest.raises(RuntimeError):
        # We expect a runtime error (or produce wrong results) because the kernel
        # assumes float pointers and uses __ldg for float loading.
        C = kernel_module.forward(A, B)
        # Make sure to synchronize to force any error.
        torch.cuda.synchronize()

# (Optional) Issue 3: Misuse of __ldg with non–contiguous input could cause issues.
def test_non_contiguous_inputs():
    N = 256
    A = torch.randn(N, N, device='cuda', dtype=torch.float32).tril()
    B = torch.randn(N, N, device='cuda', dtype=torch.float32).tril()
    # Create non-contiguous views by transposing
    A_view = A.t()
    B_view = B.t()
    # Make them lower triangular (on the original, they are, but the transpose is upper triangular)
    # So force them to be lower by taking tril of the contiguous copies.
    A_noncontig = torch.tril(A_view.clone())
    B_noncontig = torch.tril(B_view.clone())
    
    kernel_module = build_kernel()
    
    # Even though the inputs are non–default contiguous (after our manipulation),
    # the kernel expects properly laid out contiguous memory,
    # so this test may trigger issues depending on alignment.
    with pytest.raises(Exception):
        # We expect an error, incorrect results, or an assertion failure
        # because the kernel is not designed for non–contiguous input.
        C = kernel_module.forward(A_noncontig, B_noncontig)
        torch.cuda.synchronize()

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
