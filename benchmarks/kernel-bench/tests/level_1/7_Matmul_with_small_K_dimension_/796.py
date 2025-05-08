
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the kernel module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Wrong dtype is accepted though kernel expects float
def test_wrong_dtype():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    my_module = build_kernel()
    M, K, N = 64, 32, 64
    # Use double (float64) instead of float32
    A = torch.randn(M, K, device='cuda', dtype=torch.double)
    B = torch.randn(K, N, device='cuda', dtype=torch.double)
    # Even though CHECK_INPUT passes (only checks cuda and contiguity),
    # the kernel interprets the underlying bits as float.
    C_kernel = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A.to(torch.float32), B.to(torch.float32))
    # The result will be completely off since double precision data is misinterpreted.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-3), \
           "Kernel incorrectly accepted non-float32 input without error."

# Issue 2: Shape mismatch is not validated by the kernel
def test_shape_mismatch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    my_module = build_kernel()
    # Create A of shape (M, K) and B of shape (K+1, N) so that inner dimensions do not match.
    M, K, N = 64, 32, 64
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K + 1, N, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # It is expected that an out‐of‐bound memory access or similar error will occur.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 3: No error checking after kernel launch.
# We simulate an error by deliberately passing in an input that is noncontiguous.
def test_non_contiguous():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    my_module = build_kernel()
    # Create contiguous A and B then make a noncontiguous tensor
    A = torch.randn(64, 32, device='cuda', dtype=torch.float32)
    B = torch.randn(32, 64, device='cuda', dtype=torch.float32)
    A_nc = A.t()  # transpose makes it noncontiguous
    with pytest.raises(RuntimeError):
        # The CHECK_INPUT macro should trigger since A_nc is not contiguous.
        C = my_module.forward(A_nc, B)
        torch.cuda.synchronize()

# Issue 4: Loop unrolling assumption might fail when K is large.
# While this is a performance/design issue rather than a crash or error,
# we can test for correctness with a larger K than expected.
def test_large_K():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    my_module = build_kernel()
    M, K, N = 128, 512, 128  # Larger K value
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    C_kernel = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # Even if performance may suffer or registers are overused,
    # the resulting output should be numerically correct.
    assert torch.allclose(C_kernel, C_ref, atol=1e-5), \
           f"Kernel output mismatch for large K; max difference: {(C_kernel-C_ref).abs().max()}"
