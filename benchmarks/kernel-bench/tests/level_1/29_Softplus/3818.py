
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Helper to build/reload our CUDA extension.
def build_kernel():
    cuda_module = load(
        name="softplus_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Using hardcoded float literals instead of templated threshold values.
# For double inputs the branch selection can be different compared to torch.nn.functional.softplus.
def test_dtype_conversion():
    my_module = build_kernel()
    # Use double precision input (which forces scalar_t==double in the kernel)
    x = torch.randn(1024, device="cuda", dtype=torch.double) * 40.0  # scale to trigger high and low thresholds
    y_kernel = my_module.forward(x)
    y_ref = F.softplus(x)
    # Expect a discrepancy if the threshold constants are not correctly typed.
    # We require a very tight tolerance. This test should fail if the thresholds are wrong.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-8), (
        "Test did not trigger error for double precision threshold conversion issue."
    )

# Issue 2: Unnecessary use of shared memory.
# While this may not cause an outright error on small tensors, we can trigger it by stressing
# the shared memory mechanism with a tensor size smaller than the block size.
def test_small_tensor_shared_memory():
    my_module = build_kernel()
    # Create a tensor that is smaller than the thread block (so some threads in block do no load)
    x = torch.randn(128, device="cuda", dtype=torch.float32)
    y_kernel = my_module.forward(x)
    y_ref = F.softplus(x)
    # The extra load from shared memory may not be “mathematically wrong”
    # but if there is any issue in the use of shared memory then the result will differ.
    assert torch.allclose(y_kernel, y_ref, atol=1e-6), (
        "Mismatch for small tensor. Likely shared memory handling error."
    )

# Issue 3: Kernel assumes contiguous input.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a 2D tensor and take a transposed view (noncontiguous)
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    x_nc = x.t()  # transposed => noncontiguous, but same element order when flattened incorrectly
    # If the kernel always uses data_ptr and numel(), the computed softplus may use the wrong ordering
    y_kernel = my_module.forward(x_nc)
    # Use the same flattened ordering for reference
    y_ref = F.softplus(x_nc)
    # We expect the results to match if the kernel supported non-contiguous memory.
    # This test verifies that the assumption in the kernel is not general.
    assert torch.allclose(y_kernel, y_ref, atol=1e-6), (
        "Kernel output on noncontiguous input differs from reference. "
        "The kernel incorrectly assumes contiguous memory."
    )

# Issue 4: Fixed block configuration assumptions.
def test_large_tensor_fixed_grid():
    my_module = build_kernel()
    # Create a tensor with a large number of elements with a shape that is not a plain 1D vector.
    x = torch.randn(97, 1234, device="cuda", dtype=torch.float32)
    y_kernel = my_module.forward(x)
    y_ref = F.softplus(x)
    # If the kernel does not handle grid/block configuration well for arbitrary shapes,
    # the output will differ.
    assert torch.allclose(y_kernel, y_ref, atol=1e-6), (
        "Kernel output with unusual tensor shape differs from reference. "
        "Likely fixed launch configuration issue."
    )

# Issue 5: Lack of device error checking.
# We simulate an error condition by passing an input pointer that is invalid.
# (It is not easy to force the kernel to generate an error, so instead we simulate a wrong input type.)
def test_bad_input_error_check():
    my_module = build_kernel()
    # Pass in an integer tensor instead of floating point.
    x = torch.randint(0, 10, (1024,), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError):
        # The kernel is written for floating point numbers,
        # so using an integer type should trigger an error (or at least mis‐behave).
        _ = my_module.forward(x)

if __name__ == "__main__":
    pytest.main([__file__])
