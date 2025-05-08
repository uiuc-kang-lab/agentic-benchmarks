
import os
import pytest
import torch
from torch.utils.cpp_extension import load

# Function to load the compiled CUDA kernel extension
def build_kernel():
    # Ensure we are using CUDA in the extension build
    cuda_module = load(
        name="softplus_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: No error check after kernel launch.
# We'll try to force a kernel launch error by requesting an extremely large tensor.
# (If the kernel were properly checking for errors, an exception would be raised.)
def test_kernel_launch_error():
    mod = build_kernel()
    # Pick a tensor size that is huge in order to trigger a kernel launch error.
    # Note: Allocation might fail here instead, but if allocation succeeds,
    # the kernel launch should error because of an enormous grid size.
    try:
        huge_size = 2**29  # ~536 million elements, may be too high for grid configuration.
        x = torch.randn(huge_size, device="cuda", dtype=torch.float32)
    except RuntimeError:
        pytest.skip("Could not allocate huge tensor, skipping kernel launch error test.")

    with pytest.raises(RuntimeError):
        # Expecting that launching with an extremely large tensor will trigger an error;
        # if not, the lack of kernel error checking makes the bug latent.
        y = mod.forward(x)
        # Force device synchronization to expose any asynchronous errors.
        torch.cuda.synchronize()

# Issue 2: Assumes input tensor is contiguous.
# Create a noncontiguous tensor and compare the result with the correct softplus.
def test_non_contiguous_input():
    mod = build_kernel()
    # Create a contiguous tensor and a noncontiguous view (via transpose).
    x = torch.randn(16, 32, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # Transpose yields a noncontiguous tensor.
    # Compute expected result by applying softplus to a contiguous clone of the noncontiguous tensor.
    expected = torch.nn.functional.softplus(x_noncontig.contiguous()).view_as(x_noncontig)
    # Call the CUDA kernel on the noncontiguous tensor.
    result = mod.forward(x_noncontig)
    torch.cuda.synchronize()
    # Because the kernel ignores stride information, it will read the underlying memory incorrectly.
    # Thus we expect the result to differ from the correct softplus.
    assert not torch.allclose(result, torch.nn.functional.softplus(x_noncontig)), (
        "Kernel unexpectedly produced correct result on a noncontiguous tensor."
    )

# Issue 3: Does not support half precision inputs.
def test_half_precision_input():
    mod = build_kernel()
    x = torch.randn(16, 32, device="cuda", dtype=torch.float16)
    # The kernel dispatches only float and double.
    # We expect it either fails or produces wrong results.
    with pytest.raises(RuntimeError):
        y = mod.forward(x)
        torch.cuda.synchronize()

# Issue 4: No device check—the kernel does not enforce that the input is on CUDA.
def test_cpu_input():
    mod = build_kernel()
    x = torch.randn(16, 32, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        y = mod.forward(x)
        # If the kernel were run on CPU mistakenly, this would be an error.
        torch.cuda.synchronize()

# Issue 5: Deprecated usage of input.type() in AT_DISPATCH_FLOATING_TYPES.
# Since this is a compile‐time/development-time issue, we simulate a test by checking that
# the source file 'kernel.cu' still contains the deprecated API usage.
def test_deprecated_usage():
    # Ensure the file exists
    if not os.path.exists("kernel.cu"):
        pytest.skip("kernel.cu file not found; skipping deprecated usage test.")
    with open("kernel.cu", "r") as f:
        source = f.read()
    # Look for usage of "input.type(" which is deprecated. (Note: This is a simple string search.)
    assert "input.type(" in source, (
        "The kernel code does not use the deprecated input.type() API; expected deprecation for testing."
    )
