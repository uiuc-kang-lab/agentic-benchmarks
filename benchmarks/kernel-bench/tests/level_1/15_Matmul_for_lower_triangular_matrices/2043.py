
import torch
import pytest
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

# Issue 1: Floating point precision in tile index mapping.
# For very large matrices the block mapping may be computed incorrectly.
def test_large_matrix_precision():
    # Use a very large matrix size so that the number of blocks is huge.
    # Even though this may stress the GPU, we choose a size large enough to provoke potential precision issues.
    N = 16384  # Large matrix size
    # Create lower triangular matrices as required.
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    
    mod = build_kernel()
    C_kernel = mod.forward(A, B)
    torch.cuda.synchronize()
    
    # Compute reference result using PyTorch’s tril(matmul(.))
    C_ref = torch.tril(torch.matmul(A, B))
    max_diff = (C_kernel - C_ref).abs().max()
    # We set a relatively tight tolerance. Failure means the block mapping is producing wrong results.
    assert torch.allclose(C_kernel, C_ref, atol=1e-3), f"Large matrix test failed! Max diff: {max_diff}"

# Issue 2: Vectorized memory load without alignment check.
# Using a non-contiguous (mis-aligned) tensor can trigger invalid memory load.
def test_non_aligned_input():
    N = 1024
    # Create a larger contiguous tensor
    A_full = torch.tril(torch.randn(N+1, N+1, device="cuda", dtype=torch.float32))
    B_full = torch.tril(torch.randn(N+1, N+1, device="cuda", dtype=torch.float32))
    # Create non-contiguous views by slicing (which may break the assumed alignment)
    A = A_full[1:, 1:]
    B = B_full[1:, 1:]
    # Check that the tensor is indeed not contiguous.
    assert not A.is_contiguous()
    mod = build_kernel()
    # The kernel assumes contiguous float32 data and aligned to float4.
    # This test should either crash or produce an incorrect result.
    C_kernel = mod.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.tril(torch.matmul(A, B))
    max_diff = (C_kernel - C_ref).abs().max()
    assert torch.allclose(C_kernel, C_ref, atol=1e-3), f"Non-aligned input test failed! Max diff: {max_diff}"

# Issue 3: Kernel specialized for lower-triangular matrices.
# When passing full (non-lower-triangular) matrices, the kernel will compute an incorrect result.
def test_non_lower_triangular_input():
    N = 512
    # Create full (non-lower-triangular) random matrices.
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    mod = build_kernel()
    C_kernel = mod.forward(A, B)
    torch.cuda.synchronize()
    # The correct result for full matrix multiplication is different from computing tril(matmul(.))
    C_ref = torch.matmul(A, B)
    # Since the kernel zeroes out upper-triangular parts, the result should not match C_ref.
    # We test that the discrepancy is significant.
    diff = (C_kernel - C_ref).abs().max()
    assert diff > 1e-3, f"Non-lower-triangular input test did not trigger the issue! Diff: {diff}"

if __name__ == "__main__":
    pytest.main([__file__])
