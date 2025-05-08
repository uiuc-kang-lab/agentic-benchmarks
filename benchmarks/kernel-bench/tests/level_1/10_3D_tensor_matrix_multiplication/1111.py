
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build/load the CUDA extension from kernel.cu.
def build_kernel():
    return load(
        name="custom_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )

# Issue 1: Dimension mismatch test.
# The kernel does not check that A.shape[2] == B.shape(0).
# Here we deliberately create mismatched dimensions.
def test_dimension_mismatch():
    cuda_module = build_kernel()
    # Let A be of shape (N, M, K) and B be of shape (K+1, L)
    N, M, K, L = 4, 8, 16, 10
    A = torch.randn(N, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K + 1, L, device="cuda", dtype=torch.float32)
    
    # Since there's no check in the kernel, the multiplication will use the wrong stride,
    # resulting in an output that does not match torch.matmul.
    C_kernel = cuda_module.forward(A, B)
    C_ref = None
    with pytest.raises(RuntimeError):
        # torch.matmul should raise an error due to shape mismatch.
        C_ref = torch.matmul(A, B)
    
    # If by chance torch.matmul did not error (in a hypothetical scenario),
    # we check that the kernel result is incorrect.
    if C_ref is not None:
        assert not torch.allclose(C_kernel, C_ref, atol=1e-5), (
            f"Kernel output matched reference output even for incompatible dimensions."
        )

# Issue 2: Half precision handling with __ldg.
# This test uses half precision to verify that the kernel’s use of __ldg does not
# lead to accumulation inaccuracies. On devices or with improper __ldg usage for half,
# the result may significantly differ from torch.matmul.
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 6,
    reason="Half precision tests require a device with compute capability >= 6 (approx.)"
)
def test_half_precision_inaccuracy():
    cuda_module = build_kernel()
    # Use shapes that follow the expected dimensions.
    N, M, K, L = 2, 4, 32, 5
    A = torch.randn(N, M, K, device="cuda", dtype=torch.half)
    B = torch.randn(K, L, device="cuda", dtype=torch.half)
    
    C_kernel = cuda_module.forward(A, B)
    C_ref = torch.matmul(A, B)
    
    # Typically one would allow some lax tolerance with half precision,
    # but if __ldg is mishandled, the error might be significantly large.
    # Here we trigger the issue by asserting if the difference is above a very small threshold.
    if not torch.allclose(C_kernel, C_ref, atol=1e-2):
        raise AssertionError(
            f"Kernel output for half precision differs from reference output by more than tolerance. "
            f"Max difference: {(C_kernel - C_ref).abs().max().item()}"
        )

# Issue 3: Use of "#pragma unroll" on a variable loop bound.
# This test uses a non-standard K that is not a compile-time constant (e.g. 2049)
# which may trigger potential compilation/performance issues with the unrolling directive.
def test_unroll_with_variable_K():
    cuda_module = build_kernel()
    # Set K to a value that is not likely to be optimized by compile-time unrolling.
    N, M, L = 2, 4, 7
    K = 2049  # A non-standard K value to challenge the unrolling pragma.
    A = torch.randn(N, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, L, device="cuda", dtype=torch.float32)
    
    C_kernel = cuda_module.forward(A, B)
    C_ref = torch.matmul(A, B)
    
    # We allow a slightly larger tolerance here, but if the unroll directive is detrimental,
    # the kernel result may be far off.
    if not torch.allclose(C_kernel, C_ref, atol=1e-4):
        raise AssertionError(
            f"Kernel output with K={K} differs from reference output. Max difference: "
            f"{(C_kernel - C_ref).abs().max().item()}"
        )

if __name__ == "__main__":
    pytest.main([__file__])
