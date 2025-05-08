
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Helper function to build the CUDA kernel extension
def build_kernel():
    cuda_module = load(
        name="tril_mm_kernel",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Off-by-one error in B tile loading.
# We trigger this by choosing a dimension N that is not an exact multiple of TILE_SIZE.
def test_off_by_one_B_loading():
    # Choose N so that for some tile, (tile*TILE_SIZE + ty) == N (out‐of‐bound)
    N = 17  # 17 is not a multiple of 16, so the last tile will try to access index 16 (which is valid)
             # but the condition <= N could allow an index N (i.e. 17) to be read.
    # Create lower triangular matrices.
    A = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    mod = build_kernel()
    # Run the kernel
    C = mod.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference answer on CPU 
    C_ref = torch.tril(torch.matmul(A, B))
    # Test: the result may be incorrect because of off-by-one memory loads.
    assert not torch.allclose(C, C_ref, atol=1e-4), "Test did not trigger off-by-one issue: outputs unexpectedly match."

# Issue 2: Incorrect enforcement of lower-triangular property for matrix B.
# For a proper lower triangular B, the kernel should ignore contributions from B above the diagonal, 
# but since it does not check this, we can craft a test where these extra elements cause a discrepancy.
def test_incorrect_lower_triangular_B():
    N = 64  # A moderate size where tiles are fully present.
    # Create A as lower triangular.
    A = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    # Create B that is NOT lower triangular, with nonzero entries in the upper triangular part.
    B = torch.randn(N, N, device='cuda', dtype=torch.float32)
    # Force B to have nonzeros above the diagonal (even though the intended use is lower triangular)
    B = torch.tril(B) + torch.triu(B, diagonal=1)
    mod = build_kernel()
    C = mod.forward(A, B)
    torch.cuda.synchronize()
    # Reference: only use the lower triangular part of B before multiplication.
    C_ref = torch.tril(torch.matmul(A, torch.tril(B)))
    # Because the kernel does not enforce B’s triangularity properly, we expect a mismatch.
    assert not torch.allclose(C, C_ref, atol=1e-4), "Test did not trigger incorrect triangular enforcement: outputs unexpectedly match."

# Issue 3 & 4: Inadequate handling of non-multiple of TILE_SIZE sizes and mis-partitioning with streams.
# Choose an N that is not a multiple of TILE_SIZE to expose both the tiling boundary condition and the stream partitioning.
def test_non_multiple_of_tile_size_and_stream_partition():
    # Choose N such that streams and tile boundaries misalign.
    N = 50  # Not a multiple of 16.
    A = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    mod = build_kernel()
    C = mod.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.tril(torch.matmul(A, B))
    # Because of potential mis-partitioning and tile-boundary issues, the result is likely wrong.
    assert not torch.allclose(C, C_ref, atol=1e-4), "Test did not trigger tiling/stream mis-partitioning issue: outputs unexpectedly match."

if __name__ == '__main__':
    pytest.main([__file__])
