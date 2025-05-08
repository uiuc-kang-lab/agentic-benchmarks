
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build and load the CUDA extension from kernel.cu
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_dtype_float32_requirement():
    """
    Test to trigger the issue of unsupported input data types.
    The kernel assumes float32 inputs.
    This test provides float64 (double) tensors. The kernel will try to reinterpret 
    the data as float and should either raise an error (from the TORCH_CHECK in host code) or produce wrong results.
    """
    module = build_kernel()
    M, K, N = 64, 32, 64
    # Create tensors with dtype float64 instead of float32.
    A = torch.randn(M, K, device="cuda", dtype=torch.float64)
    B = torch.randn(N, K, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        # The TORCH_CHECK in the kernel host function assumes float32.
        C = module.forward(A, B)
        # Force synchronization to catch asynchronous errors.
        torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_batched_input_not_supported():
    """
    Test to trigger the issue with unsupported batched (3D) inputs.
    The kernel host function explicitly checks that A and B are 2D.
    """
    module = build_kernel()
    batch, M, K, N = 4, 16, 32, 16
    # Create batched inputs (3D) instead of the required 2D.
    A = torch.randn(batch, M, K, device="cuda", dtype=torch.float32")
    B = torch.randn(batch, N, K, device="cuda", dtype=torch.float32")
    with pytest.raises(RuntimeError):
        C = module.forward(A, B)
        torch.cuda.synchronize()
