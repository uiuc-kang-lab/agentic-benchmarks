
import pytest
import torch
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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_double_tensor():
    # Issue 1: The kernel only supports float32. Using double should yield incorrect results.
    M, K, N = 64, 128, 32
    A = torch.randn(M, K, device="cuda", dtype=torch.double)
    B = torch.randn(K, N, device="cuda", dtype=torch.double)
    mod = build_kernel()
    # The kernel will reinterpret the underlying memory as float so the result will be wrong.
    C_kernel = mod.forward(A, B)
    # Compute reference with double precision.
    C_ref = torch.matmul(A, B)
    # Cast reference to float to simulate potential kernel behavior
    C_ref_float = C_ref.float()
    # The results should not be allclose because of precision/type mismatch.
    assert not torch.allclose(C_kernel, C_ref_float, atol=1e-5), \
        "Kernel unexpectedly produced a result matching double precision input."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_noncontiguous_tensor():
    # Issue 2: The kernel requires inputs to be contiguous.
    M, K, N = 64, 128, 32
    A = torch.randn(K, M, device="cuda", dtype=torch.float32).t()  # transposed, likely non-contiguous
    B = torch.randn(N, K, device="cuda", dtype=torch.float32).t()  # transpose to get shape (K, N)
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # The TORCH_CHECK in the kernel host function should trigger an error for non-contiguous inputs.
        mod.forward(A, B)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_large_dimension_scalability():
    # Issue 3: The kernel launch configuration is very specific.
    # Although the kernel may still produce correct answers, mapping one warp per output element
    # (with grid.x = N and grid.y = ceil(M/WARPS_PER_BLOCK)) might hit hardware limits for exceedingly large N.
    # Here we simulate a moderately large case.
    M, K, N = 512, 256, 2048  # Note: these are large but should not exceed GPU grid limits on most hardware.
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    mod = build_kernel()
    C_kernel = mod.forward(A, B)
    C_ref = torch.matmul(A, B)
    # Check that the result is correct.
    assert torch.allclose(C_kernel, C_ref, atol=1e-5), "Kernel output does not match torch.matmul for large dimensions."
