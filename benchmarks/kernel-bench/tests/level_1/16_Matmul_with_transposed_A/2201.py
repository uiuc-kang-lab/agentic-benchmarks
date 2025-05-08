
import torch
import pytest
from torch.utils.cpp_extension import load
import os

def build_kernel(extra_cuda_cflags=None, define_macros=None):
    # Build the module from kernel.cu with optional extra flags and macros.
    extra_cuda_cflags = extra_cuda_cflags or ["-O3", "--use_fast_math"]
    if define_macros is not None:
        for macro, value in define_macros.items():
            extra_cuda_cflags.append(f"-D{macro}={value}")
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Unrolling assumption (BLOCK_K not multiple of 4)
def test_unroll_issue():
    # Force BLOCK_K to be 30 instead of 32.
    # This should trigger out-of-bound shared memory access in the unroll loop.
    # We expect that the result will differ from the reference.
    define_macros = {"BLOCK_K": 30}
    my_module = build_kernel(define_macros=define_macros)
    
    # Use sizes that stress the kernel.
    M = 128
    K = 128
    N = 128
    # Create input tensors in the shapes expected by the kernel:
    # A of shape (K, M) and B of shape (K, N).
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    # The expected multiplication is: C = A^T @ B.
    C_ref = torch.matmul(A.T, B)
    C = my_module.forward(A, B)
    # We intentionally check that the result is NOT close to the reference.
    # (If they compare, the issue has not been triggered.)
    assert not torch.allclose(C, C_ref, atol=1e-3), (
        "Test unroll issue: Kernel output unexpectedly matches reference even with BLOCK_K not a multiple of 4."
    )

# Issue 2: Constant memory race condition with concurrent launches in different streams.
def test_constant_memory_concurrency():
    my_module = build_kernel()
    
    M1, K1, N1 = 128, 256, 64
    M2, K2, N2 = 64, 256, 128
    # Create two sets of inputs; note that the kernel expects shape (K, M) for A and (K, N) for B.
    A1 = torch.randn(K1, M1, device="cuda", dtype=torch.float32)
    B1 = torch.randn(K1, N1, device="cuda", dtype=torch.float32)
    A2 = torch.randn(K2, M2, device="cuda", dtype=torch.float32)
    B2 = torch.randn(K2, N2, device="cuda", dtype=torch.float32)
    
    # Create two CUDA streams and launch the kernel concurrently.
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    with torch.cuda.stream(stream1):
        C1 = my_module.forward(A1, B1)
    with torch.cuda.stream(stream2):
        C2 = my_module.forward(A2, B2)
    
    # Synchronize streams.
    torch.cuda.synchronize()
    
    # Compute expected outputs.
    C1_ref = torch.matmul(A1.T, B1)
    C2_ref = torch.matmul(A2.T, B2)
    
    # Because constant memory is overwritten between kernel launches,
    # at least one of the results is likely to be wrong.
    cond1 = torch.allclose(C1, C1_ref, atol=1e-3)
    cond2 = torch.allclose(C2, C2_ref, atol=1e-3)
    assert not (cond1 and cond2), (
        "Test constant memory concurrency: Both concurrent kernel outputs match the references, "
        "which is unexpected given potential constant memory race conditions."
    )

# Issue 3: Handling non-contiguous inputs.
def test_non_contiguous_inputs():
    my_module = build_kernel()
    
    M = 128
    K = 256
    N = 64
    # Create contiguous tensors.
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # Create noncontiguous versions by transposing back and forth.
    A_noncontig = A.t().t()  # Although mathematically the same, this can be noncontiguous.
    B_noncontig = B.t().t()
    
    assert not A_noncontig.is_contiguous(), "A_noncontig is actually contiguous."
    assert not B_noncontig.is_contiguous(), "B_noncontig is actually contiguous."
    
    C = my_module.forward(A_noncontig, B_noncontig)
    C_ref = torch.matmul(A_noncontig.T, B_noncontig)
    
    # Likely the kernel will compute an incorrect result if it assumes contiguity.
    assert not torch.allclose(C, C_ref, atol=1e-3), (
        "Test non-contiguous inputs: Kernel output unexpectedly matches reference output for non-contiguous inputs."
    )

# Issue 4: Fixed tile and block sizes not matching arbitrary dimensions.
def test_inexact_tile_sizes():
    my_module = build_kernel()
    
    # Choose dimensions that are not multiples of TILE_M (16) or TILE_N (16)
    M = 70  # Not a multiple of 16.
    N = 45  # Not a multiple of 16.
    K = 128
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    C = my_module.forward(A, B)
    C_ref = torch.matmul(A.T, B)
    
    # Although the kernel includes boundary checks, subtle issues may occur.
    # To trigger the potential issue, we expect a discrepancy.
    assert not torch.allclose(C, C_ref, atol=1e-3), (
        "Test inexact tile sizes: Kernel output unexpectedly matches reference output for non-tile-multiple dimensions."
    )
