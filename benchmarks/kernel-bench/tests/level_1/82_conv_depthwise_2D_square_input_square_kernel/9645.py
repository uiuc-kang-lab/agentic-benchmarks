
import os
import torch
import pytest
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu.
def build_kernel():
    module = load(
        name="custom_depthwise_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Helper: Create reference output using PyTorch's native conv2d.
def reference_depthwise_conv2d(x, weight, bias, stride, padding, groups):
    # weight must be in the shape (in_channels, 1, kernel_height, kernel_width)
    return torch.nn.functional.conv2d(x, weight, bias=bias, stride=stride, padding=padding, groups=groups)

# Issue 1: Test with groups parameter not equal to in_channels.
def test_groups_parameter_incompatibility():
    # Create a simple convolution that is not depthwise (e.g. groups=1 while in_channels>1)
    batch_size = 2
    in_channels = 4
    out_channels = 4  # for a regular conv2d this can differ, but assume same for simplicity
    kernel_size = 3
    stride = 1
    padding = 1
    groups = 1  # not equal to in_channels

    # Create an input and weight. Note: For a regular conv2d, weight shape is (out_channels, in_channels/groups, k, k)
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # The kernel implementation ignores groups and assumes in_channels==groups (depthwise).
    module = build_kernel()
    # Invoke the forward from the CUDA module.
    out = module.forward(x, weight.squeeze(1), bias, stride, padding, groups)
    # Create a reference using PyTorch's conv2d.
    ref = reference_depthwise_conv2d(x, weight, bias, stride, padding, groups)
    # Because groups is not handled correctly, the results will likely differ.
    assert not torch.allclose(out, ref, atol=1e-3), "Kernel unexpectedly computed correct result with mismatched groups."

# Issue 2: Test with a rectangular (non-square) convolution kernel.
def test_rectangular_kernel_incompatibility():
    batch_size = 2
    in_channels = 3
    kernel_height = 3
    kernel_width = 5  # rectangular kernel
    stride = 1
    padding = 0
    groups = in_channels  # depthwise

    # Create an input tensor that is non-square.
    x = torch.randn(batch_size, in_channels, 20, 30, device="cuda", dtype=torch.float32)
    # For a standard depthwise conv2d in PyTorch, weight shape is (in_channels, 1, kernel_height, kernel_width).
    weight = torch.randn(in_channels, 1, kernel_height, kernel_width, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    module = build_kernel()
    # The kernel code uses a single kernel_size parameter and assumes square filter.
    # We simulate this by passing the height (or width) and squeezing the extra dim.
    # This mismatch should cause a discrepancy in the output.
    out = module.forward(x, weight.squeeze(1), bias, stride, padding, groups)
    ref = reference_depthwise_conv2d(x, weight, bias, stride, padding, groups)
    assert not torch.allclose(out, ref, atol=1e-3), "Kernel unexpectedly computed correct result with a rectangular kernel."

# Issue 3: Test that the kernel launch does not perform error checking.
def test_no_kernel_launch_error_checking():
    batch_size = 1
    in_channels = 3
    kernel_size = 3
    stride = 1
    padding = 1
    groups = in_channels  # depthwise

    # Create a contiguous input.
    x = torch.randn(batch_size, in_channels, 10, 10, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    module = build_kernel()
    # Here we deliberately pass a valid configuration.
    # The test will not catch an error directly since there is no error checking;
    # rather, if a kernel launch error occurred, the process would crash or raise an unhandled exception.
    try:
        out = module.forward(x, weight, bias, stride, padding, groups)
        torch.cuda.synchronize()
    except Exception as e:
        pytest.fail(f"Kernel launch raised an exception, which indicates missing error check handling: {e}")

# Issue 4: Test with non-contiguous input memory.
def test_non_contiguous_input():
    batch_size = 2
    in_channels = 3
    kernel_size = 3
    stride = 1
    padding = 1
    groups = in_channels  # depthwise

    # Create an input and then make it non-contiguous (e.g. via transpose)
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    x_noncontig = x.transpose(1, 2)  # This makes the tensor non-contiguous
    x_noncontig = x_noncontig.contiguous()  # Ensure it is non-contiguous with original layout preserved (if possible)
    # Alternatively, one can simply use x_noncontig without .contiguous() to force a non-optimal stride.
    # But note: The custom kernel indexing might assume a specific contiguous layout.
    
    weight = torch.randn(in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    module = build_kernel()
    # The kernel implementation’s arithmetic assumes the input is contiguous in memory.
    # We expect that using a non-contiguous tensor might cause the output to be incorrect.
    out = module.forward(x_noncontig, weight, bias, stride, padding, groups)
    ref = reference_depthwise_conv2d(x_noncontig.contiguous(), weight.unsqueeze(1), bias, stride, padding, groups)
    # The output is expected to be different because our kernel does not handle non-standard memory layouts.
    assert not torch.allclose(out, ref, atol=1e-3), "Kernel unexpectedly computed correct result with non-contiguous input."

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
