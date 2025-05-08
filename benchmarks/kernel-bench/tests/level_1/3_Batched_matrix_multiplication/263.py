
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build/load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="bmm_cuda_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test for input tensor type (only float32 supported)
def test_input_tensor_type():
    # Create double precision tensors (non float32)
    batch_size = 2
    m, k, n = 4, 5, 6
    A = torch.randn(batch_size, m, k, dtype=torch.double, device="cuda")
    B = torch.randn(batch_size, k, n, dtype=torch.double, device="cuda")
    kernel = build_kernel()
    
    with pytest.raises(RuntimeError) as excinfo:
        # Since the kernel is hard-coded for float,
        # passing double tensors should trigger an error/incorrect computation.
        C = kernel.forward(A, B)
        torch.cuda.synchronize()
    # We expect an error message; if not, the test will fail.
    assert "float" in str(excinfo.value).lower(), "Kernel did not fail as expected for non-float32 input."

# Issue 2: Test for non-contiguous input tensors
def test_non_contiguous_inputs():
    batch_size = 2
    m, k, n = 8, 10, 12
    # Create contiguous tensors and then make non-contiguous versions by transposing some dimensions.
    # Since our kernel assumes contiguity, the computed result might differ from torch.bmm.
    A = torch.randn(batch_size, m, k, device="cuda", dtype=torch.float32)
    B = torch.randn(batch_size, k, n, device="cuda", dtype=torch.float32)
    # Make non-contiguous versions.
    A_noncontig = A.transpose(1, 2)  # shape becomes (batch_size, k, m)
    B_noncontig = B.transpose(1, 2)  # shape becomes (batch_size, n, k)
    # To keep the same multiplication semantics, we transpose back during multiplication.
    # But we pass the non-contiguous tensors directly to the kernel.
    kernel = build_kernel()
    # Expecting that the wrong pointer arithmetic (assuming contiguity) leads to a mismatch.
    with pytest.raises(AssertionError):
        # Compute kernel output (this is expected to be wrong)
        C_kernel = kernel.forward(A_noncontig, B_noncontig)
        torch.cuda.synchronize()
        # Reference: do proper contiguous multiplication after making the tensors contiguous.
        A_contig = A_noncontig.contiguous()
        B_contig = B_noncontig.contiguous()
        C_ref = torch.bmm(A_contig, B_contig)
        # The kernel’s result is likely different.
        assert torch.allclose(C_kernel, C_ref, atol=1e-4), "Kernel produced correct result on non-contiguous input (unexpected)."

# Issue 3: Test for inefficient computation (naive loop with division/mod)
def test_large_matrix_efficiency():
    # This test does not aim to benchmark but to trigger a scenario where the inner loop is large.
    # The inefficiency might manifest as a timeout or very slow execution if the kernel is not optimized.
    # We use moderately large dimensions to simulate a heavy inner loop.
    batch_size = 4
    m, k, n = 128, 256, 128
    A = torch.randn(batch_size, m, k, device="cuda", dtype=torch.float32)
    B = torch.randn(batch_size, k, n, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    # Measure execution time using CUDA events.
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    C = kernel.forward(A, B)
    ender.record()
    torch.cuda.synchronize()
    elapsed_time = starter.elapsed_time(ender)
    # For illustration purposes we assume that an elapsed time above a threshold (e.g., 100ms) indicates inefficiency.
    # (The actual threshold may vary by hardware and load.)
    assert elapsed_time < 100, f"Kernel execution inefficient: took {elapsed_time}ms (expected < 100ms)."

# Issue 4: Test for kernel launch error checking (lack of synchronization/error query)
def test_kernel_launch_error_checking():
    # This test deliberately launches the kernel with invalid dimensions.
    # For example, setting M = 0 should normally produce an empty output, but limited error checking might let it slip.
    batch_size = 2
    m, k, n = 0, 5, 6  # Invalid/edge case parameters (empty matrix multiplication)
    A = torch.randn(batch_size, m, k, device="cuda", dtype=torch.float32)
    B = torch.randn(batch_size, k, n, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    # The kernel launch on empty output might not trigger an error,
    # but lack of error checking in the kernel can hide potential issues.
    C = kernel.forward(A, B)
    torch.cuda.synchronize()
    # The expected output is an empty tensor of shape (batch_size, m, n)
    assert C.shape == (batch_size, m, n), "Kernel did not produce expected empty output shape. Possibly missing launch error checks."
