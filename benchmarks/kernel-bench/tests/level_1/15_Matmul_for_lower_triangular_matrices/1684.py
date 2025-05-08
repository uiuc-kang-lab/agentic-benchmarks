
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Build and load the CUDA kernel module from kernel.cu
def build_kernel():
    # Ensure the kernel source file exists in the current directory.
    kernel_path = os.path.join(os.path.dirname(__file__), "kernel.cu")
    cuda_module = load(
        name="triangular_mm_module",
        sources=[kernel_path],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1:
# The kernel is implemented only for float (float32) data.
# Passing double-precision tensors should either produce wrong results or trigger an error.
def test_wrong_dtype():
    cuda_module = build_kernel()
    N = 64
    A = torch.randn(N, N, device="cuda", dtype=torch.double)
    B = torch.randn(N, N, device="cuda", dtype=torch.double)
    A = torch.tril(A)
    B = torch.tril(B)
    with pytest.raises(RuntimeError):
        # This call should fail because the kernel casts using data_ptr<float>() and the underlying type is double.
        cuda_module.forward(A, B)
        
# Test case 2:
# The kernel does not enforce contiguity. Noncontiguous input tensors might cause unexpected behaviors.
def test_non_contiguous():
    cuda_module = build_kernel()
    N = 128
    # Create a contiguous lower-triangular matrix and then create a noncontiguous slice.
    A_full = torch.randn(N+10, N+10, device="cuda", dtype=torch.float32)
    B_full = torch.randn(N+10, N+10, device="cuda", dtype=torch.float32)
    A_full = torch.tril(A_full)
    B_full = torch.tril(B_full)
    # take a slice that is likely noncontiguous
    A = A_full[:N, :N].t()  # transposition usually makes tensor noncontiguous
    B = B_full[:N, :N].t()
    assert not A.is_contiguous(), "A is contiguous, but expected noncontiguous input for testing"
    # The kernel does not check contiguity so it might produce wrong results.
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference on CPU (and then move to CUDA for fair comparison) using matmul followed by tril.
    C_ref = torch.tril(torch.matmul(A, B))
    # This test expects that the kernel output does not match due to noncontiguous memory.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel unexpectedly handled noncontiguous input correctly."

# Test case 3:
# The kernel is launched with a fixed block configuration assuming dimensions divisible by 32.
# When dimensions are not multiples of the block size, the boundary handling may be suboptimal.
def test_non_multiple_block_size():
    cuda_module = build_kernel()
    # Choose N that is not a multiple of 32.
    N = 33
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference result.
    C_ref = torch.tril(torch.matmul(A, B))
    # Depending on the misuse of warp primitives at the boundaries the result may be incorrect.
    # We trigger the bug by expecting a non-negligible difference.
    diff = (C - C_ref).abs().max()
    assert diff > 1e-3, f"Kernel output seems correct despite non-multiple-of-block-size input. Max diff: {diff}"

if __name__ == "__main__":
    pytest.main([__file__])
