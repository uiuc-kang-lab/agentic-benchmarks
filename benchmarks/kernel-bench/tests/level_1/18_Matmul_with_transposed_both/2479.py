
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

# Test 1: Use double precision to trigger the __fmaf_rn issue.
def test_double_precision():
    # Dimensions that are multiples of 32 to eliminate other issues
    M = 64
    K = 128
    N = 64
    # Create tensors with dtype double
    A = torch.randn(K, M, device="cuda", dtype=torch.double)
    B = torch.randn(N, K, device="cuda", dtype=torch.double)
    mod = build_kernel()
    # Many systems will report an error or produce incorrect results for double
    C = mod.forward(A, B)
    torch.cuda.synchronize()
    # The expected output from PyTorch must be computed as A.T @ B.T because the kernel computes:
    # C[i,j] = dot(A[tile_row, i], B[j, tile_col]) using the transposed input convention.
    C_ref = torch.matmul(A.T, B.T)
    assert torch.allclose(C, C_ref, atol=1e-5), f"Double precision test failed. Max diff: {(C - C_ref).abs().max()}"

# Test 2: Use dimensions that are not divisible by 32 to trigger boundary condition issues.
def test_non_divisible_dimensions():
    # Choose dimensions which are not multiples of 32.
    M = 50  # not divisible by 32
    K = 70  # not divisible by 32
    N = 45  # not divisible by 32
    # Use single precision.
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(N, K, device="cuda", dtype=torch.float32)
    mod = build_kernel()
    C = mod.forward(A, B)
    torch.cuda.synchronize()
    # Expected output: torch.matmul(A.T, B.T)
    C_ref = torch.matmul(A.T, B.T)
    # It is expected that incorrect boundary handling will result in a mismatch.
    assert torch.allclose(C, C_ref, atol=1e-5), f"Non-divisible dims test failed. Max diff: {(C-C_ref).abs().max()}"

# Test 3: Force a thread’s index to be misaligned for vectorized load
# We can indirectly test this by creating a scenario where the stride (M) is not a multiple
# of the vector size (vec_size), so that many threads will not fall on properly aligned addresses.
def test_alignment_issue():
    M = 63  # purposely chosen to likely break alignment assumptions (not a multiple of 4)
    K = 96
    N = 64
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(N, K, device="cuda", dtype=torch.float32)
    mod = build_kernel()
    C = mod.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A.T, B.T)
    assert torch.allclose(C, C_ref, atol=1e-5), f"Alignment test failed. Max diff: {(C-C_ref).abs().max()}"

if __name__ == '__main__':
    pytest.main([__file__])
