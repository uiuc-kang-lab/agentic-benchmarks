
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility to build and load the CUDA extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Test 1: Ensure that non-float32 inputs are rejected.
def test_input_tensor_type():
    my_module = build_kernel()
    # Create double precision tensors which should trigger a check error in the kernel.
    M, K, N = 16, 32, 16
    A = torch.randn(K, M, dtype=torch.float64, device='cuda')
    B = torch.randn(K, N, dtype=torch.float64, device='cuda')
    with pytest.raises(RuntimeError):
        my_module.forward(A, B)

# Test 2: Use input tensor shapes that are not multiples of the tile sizes.
def test_non_multiple_dimensions():
    my_module = build_kernel()
    # Choose dimensions that are not divisible by TILE_M (16) or TILE_N (16)
    M, K, N = 18, 45, 19  # non-multiple dimensions
    A = torch.randn(K, M, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference as C = A.T * B
    C_ref = torch.matmul(A.T, B)
    assert torch.allclose(C, C_ref, atol=1e-4), (
        f"Kernel output differs for non-multiple dimensions!"
    )

# Test 3: Pass batched inputs (3D tensors) to trigger the limitation
def test_batched_input():
    my_module = build_kernel()
    # Create a batch of matrices. The kernel is written for 2D only.
    batch = 4
    M, K, N = 16, 32, 16
    A = torch.randn(batch, K, M, dtype=torch.float32, device='cuda')
    B = torch.randn(batch, K, N, dtype=torch.float32, device='cuda')
    # Expect the kernel to either fail or produce incorrect results because it does not handle batches.
    with pytest.raises(RuntimeError):
        # We wrap the call expecting either an error or abnormal result.
        # Since the kernel forward does not support batched inputs, its behavior is undefined.
        my_module.forward(A, B)

# Test 4: Test with a configuration that challenges the fixed thread block assumption.
def test_fixed_block_assumption():
    my_module = build_kernel()
    # Even if dimensions are multiples of tile sizes, using a very small matrix forces extreme boundary conditions.
    M, K, N = 8, 16, 8  # smaller than TILE_M and TILE_N (16)
    A = torch.randn(K, M, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A.T, B)
    assert torch.allclose(C, C_ref, atol=1e-4), (
        f"Kernel output differs when using a small matrix which stresses block configuration!"
    )

# Test 5: Deliberately use mismatched K dimensions to trigger a dimension check.
def test_dimension_mismatch():
    my_module = build_kernel()
    M, K1, K2, N = 16, 32, 28, 16  # K dimension mismatch between A and B
    A = torch.randn(K1, M, dtype=torch.float32, device='cuda')
    B = torch.randn(K2, N, dtype=torch.float32, device='cuda')
    with pytest.raises(RuntimeError):
        my_module.forward(A, B)
