
import math
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to compile and load the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test case 1: Trigger error from incorrect per-element lower bound.
# We use lower-triangular matrices that are ones so that the correct result is known:
# For lower triangular matrices filled with 1, we have C[i,j] = (i - j + 1) for i >= j.
def test_incorrect_k_start():
    torch.manual_seed(0)
    N = 64  # Choose a size that spans multiple tiles.
    # Create fully lower-triangular matrices with ones.
    A = torch.tril(torch.ones(N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.ones(N, N, device="cuda", dtype=torch.float32))
    ref = torch.tril(torch.matmul(A, B))
    my_module = build_kernel()
    # The kernel produces output only for lower triangular region.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Expect that each element C[i,j] equals i - j + 1.
    expected = torch.zeros_like(C)
    for i in range(N):
        for j in range(i+1):
            expected[i,j] = i - j + 1
    # The difference should be very small if the kernel were correct.
    assert not torch.allclose(C, expected, atol=1e-5), \
           "Test failed to trigger the incorrect per-element lower bound issue. " \
           "Kernel returned correct result when error was expected."

# Test case 2: Trigger potential memory alignment issue.
# We create misaligned inputs by slicing the tensor. This will likely produce a tensor with a data pointer offset.
def test_unaligned_vectorized_load():
    torch.manual_seed(0)
    N = 128
    # Create a larger tensor then take a slice to force misalignment.
    A_full = torch.randn(N+1, N+1, device="cuda", dtype=torch.float32)
    B_full = torch.randn(N+1, N+1, device="cuda", dtype=torch.float32)
    # Take a slice that is still square but whose underlying data pointer might not be 16-byte aligned.
    A = torch.tril(A_full[1:, 1:])  
    B = torch.tril(B_full[1:, 1:])
    ref = torch.tril(torch.matmul(A, B))
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # If misalignment causes any issue then the result will deviate from the reference.
    assert not torch.allclose(C, ref, atol=1e-5), \
           "Test failed to trigger potential misaligned vectorized load issue. " \
           "Kernel produced correct result when error was expected due to misalignment."

# Test case 3: Trigger potential block mapping numerical precision issue.
# We use a larger matrix so that the triangular index inversion (using sqrtf) might suffer from rounding errors.
def test_block_mapping_precision():
    torch.manual_seed(0)
    N = 2048  # Use a larger matrix size.
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    ref = torch.tril(torch.matmul(A, B))
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # For a correct mapping the error should be negligible; if precision issues occur the results will differ.
    max_diff = (C - ref).abs().max().item()
    assert max_diff > 1e-3, \
           f"Test failed to trigger block mapping precision issue. " \
           f"Max difference {max_diff} is too small."

if __name__ == "__main__":
    pytest.main([__file__])
