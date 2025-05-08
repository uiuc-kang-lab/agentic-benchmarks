
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    # Ensure we are in a temporary directory to compile the module
    module = load(
        name="mean_reduce_kernel",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

@pytest.fixture(scope="module")
def cuda_module():
    mod = build_kernel()
    return mod

def test_noncontiguous_tensor(cuda_module):
    # Issue 1: Non-contiguous tensor input
    # Create a contiguous tensor and then make it non-contiguous by transposing some dimensions.
    x = torch.randn(4, 8, 16, device="cuda")
    x = x.transpose(0, 1)  # this makes x non-contiguous
    dim = 1  # reduction over dimension 1 (originally dim 0 before transpose)
    # Reference computation using torch.mean (which works correctly with non-contiguous input)
    ref = torch.mean(x, dim=dim)
    # Our kernel expects a contiguous layout with the reduction dim corresponding to the inner stride.
    # Thus, the kernel (which assumes contiguous inner dimension) is likely to produce an incorrect result.
    out = cuda_module.forward(x, dim)
    # The test should trigger the issue by detecting a discrepancy.
    assert not torch.allclose(out, ref, atol=1e-5), (
        "Kernel did not trigger a discrepancy on non-contiguous input; "
        "expected incorrect behavior due to wrong stride indexing."
    )

def test_zero_reduction_dimension(cuda_module):
    # Issue 2: Reduction dimension has size zero => division by zero.
    # Create an input tensor where the reduction dimension is of size zero.
    # For example, if we reduce over dim=0, then let that dim be 0.
    x = torch.empty((0, 10, 10), device="cuda")
    dim = 0
    # torch.mean on an empty tensor returns NaNs (or may error) depending on PyTorch version.
    # We want the CUDA kernel to trigger the division-by-zero issue.
    with pytest.raises(RuntimeError):
        # Depending on how the kernel handles division by zero, it may throw an error or produce NaN.
        # Here we assume that a CUDA error is raised.
        cuda_module.forward(x, dim)
    
def test_invalid_reduction_dimension(cuda_module):
    # Issue 3: Invalid reduction dimension.
    # For a tensor of dimension 3, specify a reduction dimension outside [0, 2].
    x = torch.randn(5, 6, 7, device="cuda")
    invalid_dim = 3  # invalid, since valid dims are 0,1,2
    with pytest.raises(IndexError):
        # The kernel function in the extension does not perform explicit bounds checks.
        # Calling torch.mean with an invalid dim would raise an error, so we expect similar behavior.
        cuda_module.forward(x, invalid_dim)
