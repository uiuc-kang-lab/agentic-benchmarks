
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Non-contiguous tensor input
def test_non_contiguous_input():
    # Create a contiguous tensor and then deliberately make it non-contiguous by transposing.
    batch, dim1, dim2 = 4, 8, 16
    # Normal shape: [batch, dim1, dim2] with reduction dim = 1.
    x = torch.randn(batch, dim1, dim2, device="cuda")
    # Transpose to get non-contiguous memory layout; now reduction dimension is not the proper one
    x_noncontig = x.transpose(1, 2)  # shape becomes [batch, dim2, dim1]
    # Our kernel is launched with a reduction dimension based on the provided integer.
    # We choose to reduce over dim 2 (which is originally dim1 before transpose) but the memory layout no longer matches.
    kernel = build_kernel()
    # We assume the kernel function is exposed as forward(input, dim) matching mean_reduce_cuda.
    with pytest.raises(AssertionError):
        # This test is designed to trigger a wrong result when using a non-contiguous input.
        # The kernel will likely produce an output that does not match torch.mean.
        y_kernel = kernel.forward(x_noncontig, 2)
        y_torch = torch.mean(x_noncontig, dim=2)
        # Because the kernel assumes a different memory layout a mismatch will occur.
        assert torch.allclose(y_kernel, y_torch, atol=1e-5), "Non-contiguous input not handled correctly."

# Issue 2: Atomic add for double precision
def test_double_precision_atomicAdd():
    # Create a small tensor using double precision. On GPUs where atomicAdd for double is not supported
    # the kernel may compile or run incorrectly.
    batch, dim1, dim2 = 4, 1024, 16  # choose dim1 (the reduction dim) large enough to force atomic kernel path.
    x = torch.randn(batch, dim1, dim2, device='cuda', dtype=torch.double)
    kernel = build_kernel()
    y_kernel = kernel.forward(x, 1)  # reduce over dim=1
    y_torch = torch.mean(x, dim=1)
    # Test that the result is close to torch.mean. On hardware where atomicAdd(double) fails this test should break.
    assert torch.allclose(y_kernel, y_torch, atol=1e-5), "Double precision atomicAdd may not be supported correctly."

# Issue 3: Zero-length reduction dimension.
def test_zero_length_reduction():
    # Create a tensor where the reduction dimension has length zero.
    # For example, a tensor of shape [3, 0, 5] reducing over dim=1.
    x = torch.randn(3, 0, 5, device="cuda")
    kernel = build_kernel()
    # The mean reduction along an empty dimension is mathematically undefined and torch.mean returns NaNs.
    # Our kernel will perform a division by zero.
    y_kernel = kernel.forward(x, 1)
    y_torch = torch.mean(x, dim=1)  # This produces NaNs.
    # We check that both outputs are NaN.
    assert torch.isnan(y_kernel).all() and torch.isnan(y_torch).all(), "Division by zero not handled as expected."

if __name__ == "__main__":
    pytest.main([__file__])
