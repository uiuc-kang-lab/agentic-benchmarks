
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from the kernel.cu file.
# We assume kernel.cu is in the same directory as this test file.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Helper: reference implementation using torch.max reduction over a given dim.
def reference_max_reduce(x, dim):
    return torch.max(x, dim=dim)[0]

# Test case 1: Non-contiguous input tensor.
# The kernel assumes a contiguous [outer, dim, inner] layout.
# Here we create a non-contiguous tensor (using transpose) and trigger incorrect behavior.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    # Create a contiguous tensor and then transpose it to break contiguity.
    # For example, start with shape [4, 8, 16] and then transpose dim0 and dim1,
    # then reduce over dimension 1 (which now is not laid out as expected).
    x = torch.randn(4, 8, 16, device="cuda")
    x = x.transpose(0, 1)  # now shape [8, 4, 16] but non-contiguous.
    # Our kernel expects the input to be interpretable as [outer, dim, inner].
    # Let’s choose dim=1 (i.e. second dimension) as the reduction dimension.
    dim = 1
    with pytest.raises(AssertionError):
        # Compare the kernel output vs. the reference.
        # Because the tensor is non-contiguous, the kernel will likely produce a wrong result.
        out_kernel = cuda_module.forward(x, dim)
        out_ref = reference_max_reduce(x, dim)
        torch.cuda.synchronize()
        # Force check that the outputs differ.
        assert torch.allclose(out_kernel, out_ref, atol=1e-5), "Kernel produced result for non-contiguous input, but it should be incorrect."

# Test case 2: Empty reduction dimension.
# Passing an input tensor where the reduction dimension size is 0 should trigger
# an out-of-bounds access in the kernel (or at least lead to bad results).
def test_empty_reduction_dim():
    cuda_module = build_kernel()
    # Create a tensor with an empty reduction dimension.
    # For instance, shape [batch, 0, features] with reduction over dimension 1.
    x = torch.randn(10, 0, 20, device="cuda")
    dim = 1
    # In PyTorch, torch.max will error out on an empty reduction.
    # Our CUDA kernel also does not check for dim_size == 0,
    # so we expect it to either produce an error or incorrect result.
    with pytest.raises(RuntimeError):
        out_kernel = cuda_module.forward(x, dim)
        torch.cuda.synchronize()

# Test case 3: Kernel launch error is not caught.
# We force an error by providing an input with an inner dimension so massive that
# calculating the grid dimensions overflows or is unreasonable.
# Note: This is more of a synthetic case and might depend on the hardware.
def test_kernel_launch_error():
    cuda_module = build_kernel()
    # Create a tensor with a huge inner dimension.
    # For practicality in this test, we simulate a scenario that would cause grid configuration issues.
    # We choose inner_size such that blocks_y ( = ceil(inner_size/threads)) becomes huge.
    batch = 1
    dim_size = 10
    # Choose an inner_size that is too large. (This may not physically allocate but is used to simulate a misconfiguration.)
    inner_size = 2**30  # over a billion elements
    shape = (batch, dim_size, inner_size)
    try:
        x = torch.randn(shape, device="cuda")
    except RuntimeError:
        pytest.skip("Unable to allocate tensor with huge inner dimension; skipping kernel launch error test.")

    dim = 1
    # We expect that launching the kernel with such a grid configuration would trigger a CUDA error.
    with pytest.raises(RuntimeError):
        out_kernel = cuda_module.forward(x, dim)
        torch.cuda.synchronize()
