
import pytest
import torch
import os
from torch.utils.cpp_extension import load

def build_kernel():
    # Ensure we compile with the proper flags.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test with a large N to trigger grid dimension problems.
def test_large_N():
    # Set up a situation where N is huge.
    # For many devices, gridDim.x cannot exceed 65535 so using N > 65535 should trigger a launch failure.
    M = 64
    N = 70000  # Likely above the hardware limit for gridDim.x
    K = 128
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # This should trigger a CUDA launch error due to an oversized grid dimension.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2: Test with noncontiguous inputs.
def test_noncontiguous_input():
    # Create a contiguous matrix and then create a noncontiguous version by transposing.
    M = 128
    N = 128
    K = 256
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    # Make A noncontiguous by transposing then transposing back (this creates a copy that is not contiguous)
    A_noncontig = A.t()
    A_noncontig = A_noncontig.t()  # Note: .t() returns a view; double transpose may preserve noncontiguity.
    assert not A_noncontig.is_contiguous(), "Test setup error: A_noncontig should be noncontiguous."
    my_module = build_kernel()
    # Even if the kernel does not check for contiguity, the address arithmetic may be wrong.
    C = my_module.forward(A_noncontig, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A_noncontig, B)
    # We expect the result to differ because the kernel assumes contiguous memory.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel should fail with noncontiguous input, but outputs match."

# Issue 3: Test with batched input (3D tensor) to check unsupported batched behavior.
def test_batched_input():
    # Create batched matrices.
    batch = 4
    M = 32
    N = 32
    K = 64
    A = torch.randn(batch, M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(batch, K, N, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel expects 2D tensors, so passing 3D tensors should trigger a size check failure.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 4: Test with unsupported datatype (e.g., half precision) to trigger a dispatch error.
def test_unsupported_dtype():
    # AT_DISPATCH_FLOATING_TYPES only handles float and double.
    M = 64
    N = 64
    K = 128
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect the dispatch macro to fail to find an implementation for float16.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
