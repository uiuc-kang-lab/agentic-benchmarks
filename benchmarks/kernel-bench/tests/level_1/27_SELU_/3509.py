
import torch
import pytest
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="selu_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def reference_selu(x: torch.Tensor) -> torch.Tensor:
    # PyTorch SELU for reference.
    return torch.selu(x)

# Test case to trigger issue 1: incorrect loop unrolling.
# We create an input that is not huge to make errors observable.
def test_loop_unrolling():
    kernel = build_kernel()
    # Create an input tensor with a size that isn't a multiple of the unroll factor.
    # For simplicity, use float32 which is supported for dispatch.
    N = 103  # Not a multiple of 4 and relatively small
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y_kernel = kernel.forward(x)
    y_ref = reference_selu(x)
    
    # The error arises because most elements are skipped.
    # We expect the kernel output to differ significantly.
    # Using assert not torch.allclose to indicate the error.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), (
        "Kernel unexpectedly produced correct results. Loop unrolling bug was not triggered."
    )

# Test case to trigger issue 2: lack of half precision support.
def test_half_precision_not_supported():
    kernel = build_kernel()
    # Create a half precision tensor. PyTorch SELU supports half, but our kernel doesn't.
    x = torch.randn(100, device="cuda", dtype=torch.float16)
    
    with pytest.raises(RuntimeError):
        # Expect the kernel to raise an error because float16 is not handled.
        kernel.forward(x)
