
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    module = load(
        name="tanh_kernel",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: The kernel only works for float32.
# Create a test case where we provide a tensor of type torch.double.
def test_non_float32_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    kernel = build_kernel()

    # Create a double tensor (float64)
    x = torch.randn(1024, device="cuda", dtype=torch.double)
    try:
        # The kernel is expected to misbehave (wrong results or crash)
        output = kernel.forward(x)
    except Exception as e:
        # If an exception occurs, that confirms that the kernel does not support non-float32 types.
        pytest.skip("Kernel does not support double input as expected: " + str(e))
    # If no exception then the output will be computed incorrectly.
    # Compare with torch.tanh result to show the incompatibility.
    reference = torch.tanh(x)
    # Use a loose tolerance because the results may be far off.
    assert not torch.allclose(output, reference, atol=1e-5), (
        "Kernel unexpectedly produced correct results for double input, but it should not."
    )

# Issue 2: The launch configuration does not handle tensors with fewer than 4 elements.
# Create a test case with a very small input tensor.
def test_small_input_tensor():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    kernel = build_kernel()

    # Create a tensor with fewer than 4 elements (e.g. 2 elements).
    x = torch.randn(2, device="cuda", dtype=torch.float32)
    output = kernel.forward(x)
    reference = torch.tanh(x)
    # The kernel may leave the output uninitialized due to zero blocks launching.
    # We check that the output does not match the expected result.
    assert not torch.allclose(output, reference, atol=1e-5), (
        "Kernel produced correct results for a small input tensor; expected issue with block configuration."
    )
