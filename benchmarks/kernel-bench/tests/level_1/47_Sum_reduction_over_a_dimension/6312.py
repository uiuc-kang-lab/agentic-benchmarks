
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to build the CUDA kernel module from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="sum_reduce_cuda_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test 1: Non-contiguous input tensor.
# Explanation: Our kernel assumes contiguous memory. This test creates a non-contiguous view.
def test_non_contiguous_input():
    kernel_module = build_kernel()
    # Create a contiguous tensor and then a non-contiguous permuted view.
    x = torch.randn(16, 256, 256, device="cuda")
    x_noncontiguous = x.permute(1, 0, 2)  # Now memory is not contiguous.
    # We choose a reduction dimension in the non-contiguous layout.
    reduce_dim = 1  # This dimension size corresponds to the original first dim.
    out_kernel = kernel_module.forward(x_noncontiguous, reduce_dim)
    out_ref = torch.sum(x_noncontiguous, dim=reduce_dim, keepdim=True)
    # Since the kernel uses a naïve index computation, the result is expected to differ.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), \
        "Kernel incorrectly handled non-contiguous tensor (results match reference unexpectedly)."

# Test 2: Missing error checking after kernel launch.
# Explanation: Without error checking, passing an invalid reduction dimension (e.g. out-of-bounds)
# can lead to undefined behavior. We trigger it by providing an invalid dim.
def test_invalid_reduce_dimension():
    kernel_module = build_kernel()
    # Create a simple contiguous tensor.
    x = torch.randn(16, 256, 256, device="cuda")
    invalid_dim = 10  # Invalid because x.dim() is 3.
    with pytest.raises(IndexError):
        # This should raise an error when trying to access sizes[10] in the kernel host code.
        kernel_module.forward(x, invalid_dim)

# Test 3: Unsupported (non-floating point) input type.
# Explanation: The kernel dispatch only covers AT_DISPATCH_FLOATING_TYPES.
def test_non_floating_input():
    kernel_module = build_kernel()
    # Create an integer tensor.
    x_int = torch.randint(0, 10, (16, 256, 256), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError):
        # The AT_DISPATCH_FLOATING_TYPES macro should cause a runtime error if an unsupported type is used.
        kernel_module.forward(x_int, 1)

# Test 4: Unsupported floating point type (half-precision).
# Explanation: Half precision (float16) is not included in AT_DISPATCH_FLOATING_TYPES and thus should raise an error.
def test_half_precision_input():
    kernel_module = build_kernel()
    x_half = torch.randn(16, 256, 256, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        kernel_module.forward(x_half, 1)
