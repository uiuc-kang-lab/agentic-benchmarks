
import pytest
import torch
from torch.utils.cpp_extension import load

# This helper compiles and loads the CUDA extension from "kernel.cu"
def build_kernel():
    return load(
        name="triangular_mm", 
        sources=["kernel.cu"], 
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: Passing a tensor of type float64 (double) should trigger a problem since the kernel is hard-coded for float.
def test_input_tensor_type_issue():
    kernel_module = build_kernel()
    N = 64
    # Create lower triangular matrices in float64 rather than float32.
    A = torch.randn(N, N, dtype=torch.float64, device="cuda")
    B = torch.randn(N, N, dtype=torch.float64, device="cuda")
    A = torch.tril(A)
    B = torch.tril(B)
    with pytest.raises(Exception):
        # Expect the kernel call to fail or produce an error because of the wrong input type.
        C = kernel_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2: Unused TILE_SIZE might indicate that the kernel was intended to handle larger tiles.
# Using a matrix whose dimensions are not a multiple of BLOCK_SIZE (16) can trigger suboptimal behavior or even wrong results.
def test_nondivisible_dimensions_issue():
    kernel_module = build_kernel()
    # Choose a dimension that is not divisible by 16
    N = 70
    A = torch.randn(N, N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, N, dtype=torch.float32, device="cuda")
    A = torch.tril(A)
    B = torch.tril(B)
    C_kernel = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute the reference result using PyTorch (which handles nondivisible dimensions correctly)
    C_ref = torch.tril(torch.matmul(A, B))
    # This test is designed to trigger potential issues: if the kernel mishandles boundaries the results will differ.
    # We expect the result to NOT match the reference.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-4), \
        f"Kernel output unexpectedly matches the reference on nondivisible dimensions; potential misuse of tiling/TILE_SIZE."

# Issue 3: The inner loop boundary condition in the kernel might be too strict in general cases.
# This test will use an input size that causes the tile splitting to be non-uniform (e.g. one block partially covers the summation range).
def test_inner_loop_boundary_issue():
    kernel_module = build_kernel()
    # Choose a size that forces the last tile to be partially filled
    N = 17  # 17 is not a multiple of 16 (BLOCK_SIZE)
    A = torch.randn(N, N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, N, dtype=torch.float32, device="cuda")
    # Ensure A and B are lower triangular
    A = torch.tril(A)
    B = torch.tril(B)
    C_kernel = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.tril(torch.matmul(A, B))
    # If the inner loop boundary handling is too strict the computed result will differ.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-4), \
        f"Kernel output unexpectedly matches the reference on borderline tile cases; inner loop boundary conditions may be mishandled."
