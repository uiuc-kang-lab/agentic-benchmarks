
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Ensure we rebuild the module each time (for testing)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    module = load(
        name="test_module",
        sources=[os.path.join(this_dir, "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Test 1. Non-contiguous input tensor.
def test_non_contiguous():
    my_module = build_kernel()
    # Create contiguous tensors and then make them non-contiguous by transposing a second time.
    # Note: The kernel expects A of shape (K, M) and B of shape (N, K)
    M, K, N = 37, 41, 29  # use dimensions not multiples of tile sizes to test boundary as well
    A = torch.randn(K, M, device="cuda", dtype=torch.float32).t()  # A.t() becomes non-contiguous
    A = A.t()  # make it non-contiguous again (restore original shape but not contiguous)
    B = torch.randn(N, K, device="cuda", dtype=torch.float32)
    # Expect that using a non-contiguous tensor may lead to a wrong result.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Reference: use the same multiplication method as expected from the kernel:
    # Expected computation: C[m,n] = sum_{k} A[k, m] * B[n, k]
    C_ref = torch.matmul(A.t(), B.t()).t()  # Rearranging to get C[m, n]= sum_k A[k,m]*B[n,k]
    with pytest.raises(AssertionError):
        assert torch.allclose(C, C_ref, atol=1e-5), \
            f"Kernel output matches reference even with non-contiguous input!"

# Test 2. Dimensions not multiples of block/tile sizes.
def test_incorrect_dimensions():
    my_module = build_kernel()
    # Use dimensions that are not divisible by BLOCK_SIZE_M (32) or BLOCK_SIZE_N (16)
    M, K, N = 45, 73, 27  # arbitrary dimensions not multiples of 32 or 16.
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(N, K, device="cuda", dtype=torch.float32)
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Expected: C[m,n] = sum_{k} A[k,m]*B[n,k]
    C_ref = torch.matmul(A.t(), B.t()).t()
    with pytest.raises(AssertionError):
        assert torch.allclose(C, C_ref, atol=1e-5), \
            f"Kernel output should differ when dimensions are not multiples of tile sizes! " \
            f"Difference: {(C - C_ref).abs().max()}"

# Test 3. Batched input.
def test_batched_input():
    my_module = build_kernel()
    # The kernel is designed for 2D matrices only.
    batch = 4
    M, K, N = 64, 50, 32
    A = torch.randn(batch, K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(batch, N, K, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # Expect an error because the kernel's pointer arithmetic and dimensions assume 2D.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Test 4. Unsupported data type (e.g., half precision).
def test_half_precision():
    my_module = build_kernel()
    M, K, N = 64, 50, 32
    A = torch.randn(K, M, device="cuda", dtype=torch.float16)
    B = torch.randn(N, K, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # Expect an error because AT_DISPATCH_FLOATING_TYPES does not cover half.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
