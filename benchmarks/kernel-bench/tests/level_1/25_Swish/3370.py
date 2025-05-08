
import pytest
import torch
import math
from torch.utils.cpp_extension import load

# Utility function to build the CUDA extension module
def build_kernel():
    cuda_module = load(
        name="swish_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Reference swish function on CPU using float32
def reference_swish(x: torch.Tensor) -> torch.Tensor:
    return x * (1.0 / (1.0 + torch.exp(-x)))

# Test 1: Trigger the constant naming issue.
# The kernel uses a constant named ONE_HALF that is set to 1.0f but is misnamed.
# Although the computation works as 1.0/(1.0+exp(-x)) (i.e. correct sigmoid),
# the naming mismatch might lead to mistaken modifications in more general kernels.
# Here we simply check that the kernel returns values equal to the reference.
def test_constant_naming_issue():
    device = "cuda"
    x = torch.randn(1024, device=device, dtype=torch.float32)
    module = build_kernel()
    y_kernel = module.forward(x)
    y_ref = reference_swish(x)
    # Allow a small numerical tolerance.
    assert torch.allclose(y_kernel, y_ref, atol=1e-5), \
      f"Kernel computation with constant value error: max diff {torch.abs(y_kernel - y_ref).max()}"

# Test 2: Trigger the type issue.
# Provide a double tensor and check that the result is wrong (or raises error)
def test_input_tensor_type_issue():
    device = "cuda"
    # Create a double precision tensor.
    x = torch.randn(1024, device=device, dtype=torch.float64)
    module = build_kernel()
    # The kernel expects float32. If it does not raise an error, the output will be incorrect.
    y_kernel = module.forward(x.float())  # Correct way would be to cast before calling, but we simulate wrong usage.
    # Instead, we purposely pass a double tensor by reinterpreting its data pointer (simulate misuse)
    # WARNING: This is undefined behavior. For testing purposes, we mimic the misuse by reinterpreting.
    # Create a fake double view on the same data block.
    x_double_view = x
    # Deliberately bypass the correct casting. Since our extension 
    # always uses data_ptr<float>(), we simulate a misuse by calling with a double tensor.
    # We expect the output not to match the reference when the user forgets to cast.
    y_kernel_wrong = module.forward(x_double_view)  # This uses x_double_view.data_ptr<float>()
    y_ref = reference_swish(x_double_view.float())
    # In a correct implementation, a type check would prevent this; here we expect a mismatch.
    diff = torch.abs(y_kernel_wrong - y_ref).max().item()
    assert diff > 1e-3, f"Kernel incorrectly processed a double tensor; diff {diff} is too small."

# Test 3: Trigger the issue with noncontiguous input.
# Provide a noncontiguous tensor and verify that the kernel output does not match the expected value.
def test_noncontiguous_input_issue():
    device = "cuda"
    # Create a contiguous tensor and then a noncontiguous view by transposing.
    x = torch.randn(128, 128, device=device, dtype=torch.float32)
    x_noncontig = x.t()  # Transpose makes it noncontiguous
    module = build_kernel()
    y_kernel = module.forward(x_noncontig)
    y_ref = reference_swish(x_noncontig)
    # Because the kernel does not handle noncontiguous strides correctly,
    # the output is expected to differ.
    diff = torch.abs(y_kernel - y_ref).max().item()
    assert diff > 1e-3, f"Kernel output unexpectedly matches reference for noncontiguous input; diff {diff}"

# Test 4: Trigger the lack of kernel launch error checking.
# Provide a case that forces an invalid launch configuration.
# Here we pass an input with 0 elements which might lead to a kernel launch with 0 blocks.
def test_kernel_launch_error_checking():
    device = "cuda"
    x = torch.empty(0, device=device, dtype=torch.float32)
    module = build_kernel()
    # This should safely return an empty tensor. If there was a launch error, torch.cuda.synchronize()
    # would raise an exception.
    y_kernel = module.forward(x)
    torch.cuda.synchronize()
    # Check that output is also empty.
    assert y_kernel.numel() == 0, "Kernel did not correctly handle an empty input tensor."

# Test 5: Trigger the issue of hard-coded launch configuration.
# Run the kernel on a very large tensor (e.g., exceeding the clamped block count might cause suboptimal behavior).
# While the kernel may compute correct values, the fixed 576 block limit is not general.
def test_launch_configuration_issue():
    device = "cuda"
    # Create a large tensor.
    n = 10**6  # large number of elements
    x = torch.randn(n, device=device, dtype=torch.float32)
    module = build_kernel()
    y_kernel = module.forward(x)
    y_ref = reference_swish(x)
    # Verify correctness even though performance might suffer on devices with different SM counts.
    assert torch.allclose(y_kernel, y_ref, atol=1e-5), \
      "Kernel computed incorrect swish on a large tensor, possibly due to fixed launch configuration."

if __name__ == "__main__":
    pytest.main([__file__])
