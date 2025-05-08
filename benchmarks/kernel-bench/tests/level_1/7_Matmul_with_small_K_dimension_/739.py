
import torch
import pytest
from torch.utils.cpp_extension import load

# Build/load the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel only supports float32. Test with double precision input.
def test_input_tensor_type():
    my_module = build_kernel()
    M, K, N = 64, 32, 64
    # Using double type instead of float32.
    A = torch.randn(M, K, device="cuda", dtype=torch.double)
    B = torch.randn(K, N, device="cuda", dtype=torch.double)
    with pytest.raises(RuntimeError):
        # This should fail because the kernel calls data_ptr<float>()
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2: The manual unrolling assumes TILE_DIM is a multiple of UNROLL_FACTOR.
# Although the current constants satisfy this, a non-divisible tile load (or K relative to TILE_DIM)
# might reveal issues. Here we test with K that is not a multiple of TILE_DIM.
def test_incompatible_unroll_factor():
    my_module = build_kernel()
    # Use a K dimension that is not a multiple of TILE_DIM (TILE_DIM=16, pick K=18)
    M, K, N = 64, 18, 64
    # Use float tensors so the kernel runs (even though unrolling assumptions may be violated)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # The output may be inaccurate due to unroll assumption mismatch.
    assert not torch.allclose(C, C_ref, atol=1e-3), (
        "Test expected an error due to unroll factor assumption issue, but got matching outputs."
    )

# Issue 3: The block configuration uses BLOCK_ROWS=8 while the tile is 16x16.
# This test uses matrix dimensions that are not multiples of the tile dimensions (e.g. 33)
# to trigger potential errors in shared memory loading.
def test_incorrect_shared_memory_loading():
    my_module = build_kernel()
    # Choose dimensions that force partial tiles in the last block.
    M, K, N = 33, 32, 33
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # Expect differences because not all shared memory elements are loaded correctly.
    assert not torch.allclose(C, C_ref, atol=1e-3), (
        "Test expected mismatch due to incomplete tile loading, but outputs are similar."
    )

# Issue 4: The cuBLAS fallback call does not compensate for row-major layout.
# When matrix dimensions exceed MATRIX_SIZE_THRESHOLD (512), the fallback branch is used.
def test_cublas_row_major_assumption():
    my_module = build_kernel()
    # Using large matrices to force the cuBLAS branch.
    M, K, N = 1024, 32, 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # Due to the row-major/column-major mismatch, the result from cuBLAS will be incorrect.
    assert not torch.allclose(C, C_ref, atol=1e-3), (
        "Test expected a mismatch due to wrong cuBLAS usage with row-major tensors, but outputs match."
    )
