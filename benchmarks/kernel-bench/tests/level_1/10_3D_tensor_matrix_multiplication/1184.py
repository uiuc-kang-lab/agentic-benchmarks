
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

# Test case to expose that the intended warp-level reduction is not being used.
# While this isn’t a runtime “bug” per se, we can detect its impact by comparing 
# results (or potential performance differences) with torch.matmul on inputs that 
# would benefit from warp‐level reductions (e.g. large inner dimensions).
def test_warp_level_reduction_effect():
    # Use a moderately sized tensor so that tiling matters.
    N = 4
    M = 128
    K = 256
    L = 128
    A = torch.randn(N, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, L, device="cuda", dtype=torch.float32)
    
    my_module = build_kernel()
    # Invoke our custom kernel
    C_kernel = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Reference result via torch.matmul
    C_ref = torch.matmul(A, B)
    # The error tolerance here can be small because both use fp32,
    # but differences in reduction order might introduce slight numerical differences.
    diff = (C_kernel - C_ref).abs().max().item()
    # If warp-level reductions were applied properly, we might expect less error.
    # Because our kernel does not use the warpReduceSum, the resulting reduction order is different.
    # We check that the maximum difference still remains within a reasonable tolerance.
    # (In a more optimized version one might expect lower variance.)
    assert diff < 1e-4, f"Max difference {diff} is too high, indicating potential warp-level reduction issues."

# Test case to trigger potential half-precision issues:
# The kernel dispatches for half even though it does not employ specialized __half arithmetic.
# On GPUs with proper half support torch.matmul may use tensor cores and/or fp32 accumulation,
# so the kernel result may be noticeably different.
def test_half_precision():
    N = 2
    M = 31   # purposely non-divisible by BLOCK_SIZE (16)
    K = 47   # purposely non-divisible by BLOCK_SIZE (16)
    L = 29   # purposely non-divisible by BLOCK_SIZE (16)
    A = torch.randn(N, M, K, device="cuda", dtype=torch.half)
    B = torch.randn(K, L, device="cuda", dtype=torch.half)
    
    my_module = build_kernel()
    C_kernel = my_module.forward(A, B)
    torch.cuda.synchronize()
    
    # Using torch.matmul with half typically promotes accumulation to fp32.
    # Our kernel in half mode does not do that and may be less accurate.
    C_ref = torch.matmul(A, B)
    # Convert our kernel result to float32 for a more meaningful comparison.
    diff = (C_kernel.float() - C_ref.float()).abs().max().item()
    # We set a looser tolerance here due to potential precision issues.
    assert diff < 1e-2, f"Half precision output mismatch: max diff {diff}"

if __name__ == "__main__":
    pytest.main([__file__])
