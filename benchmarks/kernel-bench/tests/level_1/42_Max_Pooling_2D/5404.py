
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper: Compile and load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="maxpool_cuda_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: The kernel is not type‐generic and uses an initialization value that only makes sense for floats.
# This test passes an integer type (torch.int32) tensor as input.
# We expect the kernel to either raise an error during dispatch or produce an incorrect result.
def test_integer_input():
    cuda_module = build_kernel()
    # Create an integer tensor as input.
    batch_size, channels, height, width = 2, 3, 8, 8
    # Use small numbers so that max pooling is defined.
    x_int = torch.randint(low=0, high=100, size=(batch_size, channels, height, width), dtype=torch.int32, device="cuda")
    # For the purpose of the test, set kernel parameters.
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    # The forward call may compile but yield wrong output,
    # so if the resulting tensor is not of type float we consider it an issue.
    with pytest.raises(Exception):
        # We expect an error due to the use of std::numeric_limits<T>::infinity() on an integer type.
        output = cuda_module.forward(x_int, kernel_size, stride, padding, dilation)
        # Optionally, one can check if output is not as expected.
        # For example, compare with PyTorch CPU version.
        expected = torch.nn.functional.max_pool2d(x_int.float(), kernel_size=kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation)
        # The test should not reach here.
        assert not torch.allclose(output.float(), expected), "Kernel unexpectedly produced a correct result for integer input."

# Issue 2: The use of unqualified max in device code may cause incorrect comparisons on floating point inputs.
# We provide a carefully chosen input where we know what the result should be and then compare it with PyTorch's own max_pool2d.
def test_incorrect_max_computation():
    cuda_module = build_kernel()
    # Create an input tensor with a pattern that challenges the maximum-reduction.
    # We use negative values to stress the initialization with -infinity.
    batch_size, channels, height, width = 1, 1, 6, 6
    # Build a tensor that has one very high number in a known location.
    x = -torch.ones(batch_size, channels, height, width, dtype=torch.float32, device="cuda") * 100.0
    # Create a pattern: set one element in each pooling window to 50, except in one window we set a larger value.
    # With kernel_size=2, stride=2, expect 3x3 output.
    x[0, 0, 1, 1] = 50.0  # a normal high value in one window
    x[0, 0, 3, 3] = 200.0  # this should become the max in its window
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    # Run custom kernel
    output = cuda_module.forward(x, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()
    # Compute expected output via PyTorch's CPU max pooling.
    expected = torch.nn.functional.max_pool2d(x, kernel_size=kernel_size,
                                               stride=stride, padding=padding, dilation=dilation)
    # If the max function in the CUDA code is not invoked correctly, the output may be different.
    # Check if the results are close; if they are (unexpectedly) correct, then the issue is not triggered.
    # We assert that the outputs are different to catch the (mis-)implementation.
    assert not torch.allclose(output, expected, atol=1e-5), (
        "Custom kernel output unexpectedly matches the expected output. "
        "This may indicate that the device max() function resolved correctly and the issue is not triggered."
    )
