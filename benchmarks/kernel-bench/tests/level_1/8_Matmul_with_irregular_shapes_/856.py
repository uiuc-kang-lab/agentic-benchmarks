
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Helper function to load the CUDA extension from kernel.cu.
def build_kernel():
    # Assume kernel.cu is in the same directory as this test file.
    module = load(
        name="test_kernel",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Issue 1: Use of constexpr function (select_tile_dim) with runtime values.
# Use a shape that is “irregular” (so that the min is not something obvious)
# and force the kernel to pick TILE_DIM based on a non-constant input.
def test_constexpr_tile_dim():
    # Using sizes that are not multiples of 16
    M, K, N = 23, 37, 41  # irregular sizes so that std::min({M,K,N}) == 23 triggers tile selection
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
    kernel = build_kernel()
    C_ker = kernel.forward(A, B)
    C_ref = torch.matmul(A, B)
    # Expect difference because the runtime computation of tile dims can lead 
    # to an incorrect tiling of boundaries.
    assert not torch.allclose(C_ker, C_ref, atol=1e-3), \
        "Test for constexpr tile dim issue did not trigger failure (unexpected match)."

# Issue 2: Kernel only supports float32. Using a different datatype (double) should fail or produce wrong results.
def test_input_tensor_type():
    M, K, N = 64, 64, 64
    A = torch.randn(M, K, device="cuda", dtype=torch.float64).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float64).contiguous()
    kernel = build_kernel()
    # The kernel does not check dtype; it will reinterpret the data as float32.
    C_ker = kernel.forward(A, B)
    C_ref = torch.matmul(A, B)
    # Because of type misinterpretation the results will be off.
    diff = (C_ker - C_ref).abs().max().item()
    assert diff > 1e-1, f"Double-precision input did not trigger type issue; max diff {diff}"

# Issue 3: Hard-coded 2x2 unrolling may fail for inputs where dimensions are not divisible by TILE_DIM.
def test_non_divisible_dimensions():
    # Choose dimensions that are barely larger than TILE_DIM (=16 or 32) so that the tile does not evenly cover the output.
    M, K, N = 17, 19, 23  # with select_tile_dim likely returning 16
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
    kernel = build_kernel()
    C_ker = kernel.forward(A, B)
    C_ref = torch.matmul(A, B)
    diff = (C_ker - C_ref).abs().max().item()
    assert diff > 1e-3, f"Non-divisible dimensions did not trigger issue; max diff {diff}"

# Issue 4: When dimensions >= CUBLAS_THRESHOLD, the fallback uses cublasSgemm.
# The use of cublasSgemm with swapped arguments may produce a transposed result.
def test_cublas_order_issue():
    # Use large matrices (all dims >= CUBLAS_THRESHOLD = 512) so that the cublasSgemm branch is taken.
    M, K, N = 600, 600, 600
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
    kernel = build_kernel()
    C_ker = kernel.forward(A, B)
    C_ref = torch.matmul(A, B)
    # If the cublas call is incorrect, the output will likely be almost the transpose of what we want.
    diff = (C_ker - C_ref).abs().max().item()
    assert diff > 1e-3, f"cublasSgemm did not trigger order/transposition issue; max diff {diff}"

if __name__ == '__main__':
    pytest.main([__file__])
