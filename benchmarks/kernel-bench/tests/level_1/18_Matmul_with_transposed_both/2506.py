
import torch
import pytest
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Force rebuild each time to capture potential issues.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_file = os.path.join(this_dir, "kernel.cu")
    cuda_module = load(
        name="test_module",
        sources=[cuda_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_incorrect_input_layout():
    """
    Issue 1: The kernel assumes that input tensors have specific layouts and are contiguous.
    Here we purposefully pass non-contiguous inputs (or wrong shape) so that the assumptions are broken.
    Expected: The result does not match torch.matmul(A.T, B.T)
    """
    # Create tensors with proper numerical values but then make them non-contiguous
    M, K, N = 512, 1023, 256  # K not multiple of TILE_SIZE (32) and non-contiguous layout
    # The kernel expects: A shape = (K, M) and B shape = (N, K)
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(N, K, device="cuda", dtype=torch.float32)
    # Make A non-contiguous by transposing it back and forth
    A_non_contig = A.T.T
    B_non_contig = B.index_select(0, torch.randperm(B.size(0)).to("cuda")).index_select(0, torch.argsort(torch.randperm(B.size(0)).to("cuda")))
    
    my_module = build_kernel()
    C = my_module.forward(A_non_contig, B_non_contig)
    torch.cuda.synchronize()
    # Compute reference result using the expected interpretation:
    C_ref = torch.matmul(A_non_contig.T, B_non_contig.T)
    # These tensors are expected to differ because the kernel assumes contiguous data with specific layout.
    assert not torch.allclose(C, C_ref, atol=1e-4), \
        f"Test for input layout did not trigger an error: difference max {(C - C_ref).abs().max()}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_non_divisible_dimensions():
    """
    Issue 2: The kernel's tiling assumes dimensions (especially K) are multiples of TILE_SIZE=32.
    Here we use dimensions that are not multiples of 32 so that boundary conditions are exercised.
    Expected: The kernel yields an incorrect result.
    """
    # Dimensions not divisible by 32:
    M, K, N = 37, 45, 29  
    # According to the kernel,
    # A should be (K, M) and B should be (N, K)
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(N, K, device="cuda", dtype=torch.float32)
    
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference result: note that forward in Python does torch.matmul(A.T, B.T)
    C_ref = torch.matmul(A.T, B.T)
    # With non-divisible dims the tiled boundary handling of the kernel is likely to be buggy.
    # Thus the result should differ.
    assert not torch.allclose(C, C_ref, atol=1e-4), \
        f"Kernel unexpectedly produced correct results with non-divisible dimensions. Max diff: {(C-C_ref).abs().max()}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_unsupported_tensor_dtype():
    """
    Issue 4: The kernel dispatch uses AT_DISPATCH_FLOATING_TYPES with A.type() which may not correctly handle
    dtypes like float64 on some devices/configurations. Here we pass double tensors.
    Expected: The result is either wrong or the kernel raises an unexpected error.
    """
    M, K, N = 256, 512, 128
    A = torch.randn(K, M, device="cuda", dtype=torch.float64)
    B = torch.randn(N, K, device="cuda", dtype=torch.float64)
    
    my_module = build_kernel()
    # We wrap the call in a try/except; if the kernel runs, we compare the results.
    try:
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
        C_ref = torch.matmul(A.T, B.T)
        # We expect the kernel, which was not really designed for float64, to produce a wrong result.
        assert not torch.allclose(C, C_ref, atol=1e-6), \
            f"Kernel unexpectedly produced correct results with float64. Max diff: {(C-C_ref).abs().max()}"
    except RuntimeError as e:
        # An error is also an acceptable outcome given the type mismatch.
        print("Caught expected RuntimeError for float64 inputs:", e)
