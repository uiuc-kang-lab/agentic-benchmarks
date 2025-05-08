
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Non-contiguous input tensors are not handled.
def test_non_contiguous():
    # Create contiguous inputs
    M, K, N = 128, 256, 64
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # Make a noncontiguous copy (e.g. by transposing or slicing)
    A_noncontig = A.t()  # Now shape is (M,K) and noncontiguous relative to the expected layout.
    B_noncontig = B.clone()  # Keep B contiguous.
    
    kernel = build_kernel()
    with pytest.raises(Exception):
        # The kernel’s indexing arithmetic will assume contiguity.
        C = kernel.forward(A_noncontig, B_noncontig)
        torch.cuda.synchronize()

# Issue 2: Batched or higher-dimensional tensors are not supported.
def test_batched_input():
    M, K, N, batch = 64, 128, 32, 10
    # Create batched tensors by adding an extra dimension.
    A = torch.randn(batch, K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(batch, K, N, device="cuda", dtype=torch.float32)
    
    kernel = build_kernel()
    with pytest.raises(IndexError):
        # The kernel will interpret the batch dimension as part of the matrix dimensions.
        C = kernel.forward(A, B)
        torch.cuda.synchronize()

# Issue 3: Only float32 inputs are accepted.
def test_wrong_dtype():
    M, K, N = 128, 256, 64
    A = torch.randn(K, M, device="cuda", dtype=torch.float64)  # double precision
    B = torch.randn(K, N, device="cuda", dtype=torch.float64)
    
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # The TORCH_CHECK in the kernel should raise an error about dtype.
        C = kernel.forward(A, B)
        torch.cuda.synchronize()

# Issue 4: Lack of full error synchronization may hide asynchronous errors.
def test_async_error_detection():
    # Provide intentionally malformed sizes so that the kernel accesses out of bounds.
    # For example, set B with incorrect dimensions.
    M, K, N_correct, N_wrong = 128, 256, 64, 32
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N_wrong, device="cuda", dtype=torch.float32)  # Wrong dimension, should be N_correct.
    
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # Although the check on dimensions is done in CPU code for K mismatch,
        # if not, the kernel global memory access can produce asynchronous errors.
        C = kernel.forward(A, B)
        # Force synchronization to surface errors.
        torch.cuda.synchronize()
