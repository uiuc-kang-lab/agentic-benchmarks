
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build and load the CUDA extension
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

@pytest.fixture(autouse=True)
def check_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

# Issue 1: Inflexible thread block and tile size configuration.
# Test with matrix dimensions that are not multiples of the tile sizes.
def test_boundary_conditions():
    # Using dimensions that are not divisible by BLOCK_SIZE_M (32), BLOCK_SIZE_N (16)
    # or BLOCK_SIZE_K (16)
    # In the kernel, A is expected to have shape (K x M) and B shape (N x K)
    M = 37   # Not a multiple of 32
    K = 45   # Not a multiple of 16
    N = 29   # Not a multiple of 16

    # Create inputs with the expected shapes
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(N, K, device="cuda", dtype=torch.float32)
    kernel_module = build_kernel()
    
    # Perform kernel multiplication: computes C[m, n] = sum_k A[k, m] * B[n, k]
    C_kernel = kernel_module.forward(A, B)
    # Compute reference result using the same computation order:
    # C[m,n] = sum_k A[k, m] * B[n, k]  which is equivalent to (A^T * B^T)[m,n]
    C_ref = torch.matmul(A.T, B.T)
    torch.cuda.synchronize()
    assert torch.allclose(C_kernel, C_ref, atol=1e-3), f"Kernel output differs from reference for non-multiple dimensions. Max diff: {(C_kernel - C_ref).abs().max()}"

# Issue 2: Reliance on __ldg and assumption of contiguous and aligned input.
def test_non_contiguous_input():
    M, K, N = 1024, 4096, 2048
    # Create contiguous inputs first
    A_contig = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B_contig = torch.randn(N, K, device="cuda", dtype=torch.float32)
    # Create non-contiguous version by transposing (note: the kernel expects A of shape (KxM) and B of shape (NxK))
    # So we simulate a scenario in which the caller might inadvertently supply noncontiguous tensors.
    A_noncontig = A_contig.T.T  # This trick forces non-contiguity sometimes. For a more robust test, we can use .clone() after a transpose.
    B_noncontig = B_contig.T.T
    
    # Ensure they are not contiguous. If they happen to be contiguous, force non-contiguity:
    if A_noncontig.is_contiguous():
        A_noncontig = A_noncontig.clone().t().t()
    if B_noncontig.is_contiguous():
        B_noncontig = B_noncontig.clone().t().t()
    
    kernel_module = build_kernel()
    C_kernel = kernel_module.forward(A_noncontig, B_noncontig)
    C_ref = torch.matmul(A_noncontig.T, B_noncontig.T)
    torch.cuda.synchronize()
    assert torch.allclose(C_kernel, C_ref, atol=1e-3), f"Kernel output differs on non-contiguous input. Max diff: {(C_kernel - C_ref).abs().max()}"

# Issue 3: Warp divergence and redundant work in boundary tiles.
# Although the kernel masks out-of-bound loads by checking indices, such conditionals can lead to warp divergence.
# This test uses small matrices that force many threads to be inactive in the tile.
def test_small_matrix():
    # Choose small dimensions so that many threads in a block will compute outside the active area.
    # A: shape (K x M), B: shape (N x K)
    M = 20   # less than BLOCK_SIZE_M (32)
    K = 10   # less than BLOCK_SIZE_K (16)
    N = 12   # less than BLOCK_SIZE_N (16)
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(N, K, device="cuda", dtype=torch.float32)
    kernel_module = build_kernel()

    C_kernel = kernel_module.forward(A, B)
    C_ref = torch.matmul(A.T, B.T)
    torch.cuda.synchronize()
    assert torch.allclose(C_kernel, C_ref, atol=1e-3), f"Kernel output differs on small matrices. Max diff: {(C_kernel - C_ref).abs().max()}"
