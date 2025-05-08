
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose_extension",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: The kernel only supports float32 tensors.
def test_non_float_input():
    cuda_module = build_kernel()
    # Create double-precision tensors intentionally
    batch_size = 4
    in_channels = 4
    input_length = 20
    kernel_size = 3
    out_channels = 8
    stride = 2
    padding = 1
    dilation = 1

    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float64)
    # Weight with shape (in_channels, out_channels, kernel_size)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float64)
    # Bias is optional: here we pass one in double.
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float64)

    # Expect the kernel to fail because it internally uses float pointers.
    with pytest.raises(RuntimeError):
        out = cuda_module.forward(x, weight, bias, stride, padding, dilation)
        torch.cuda.synchronize()  # ensure kernel error is reported

# Issue 2: No parameter validation for output length.
def test_negative_output_length():
    cuda_module = build_kernel()
    # Choose parameters such that the computed output_length becomes <= 0.
    # Using the formula:
    #   output_length = (input_length - 1)*stride - 2*padding + dilation*(kernel_size - 1) + 1
    # Let input_length=1, stride=1, padding=10, dilation=1, kernel_size=3:
    #   output_length = (0)*1 - 20 + 2 + 1 = -17
    batch_size = 2
    in_channels = 3
    input_length = 1
    kernel_size = 3
    out_channels = 5
    stride = 1
    padding = 10
    dilation = 1

    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = None

    # The kernel will compute a negative output_length, and torch::zeros will error.
    with pytest.raises(RuntimeError):
        out = cuda_module.forward(x, weight, bias, stride, padding, dilation)
        torch.cuda.synchronize()

# Issue 3: The kernel does not support grouped convolutions.
def test_group_convolution():
    cuda_module = build_kernel()
    # We simulate a grouped convolution by providing a weight tensor whose shape corresponds to a grouped conv.
    # For a grouped conv with groups=2, PyTorch's native conv_transpose1d expects weight shape:
    #   (in_channels, out_channels_per_group, kernel_size)
    # and produces an output with out_channels = groups * (out_channels_per_group).
    # The custom kernel, however, assumes weight is laid out as (in_channels, out_channels, kernel_size).
    # Therefore, if we pass the weight tensor meant for a group convolution to the custom kernel,
    # the output shape will be different from the native PyTorch version.
    batch_size = 2
    in_channels = 4
    groups = 2
    # Let out_channels (native view) be groups * out_channels_per_group; choose out_channels_per_group = 6.
    out_channels = groups * 6
    kernel_size = 3
    input_length = 15
    stride = 2
    padding = 1
    dilation = 1

    # Native grouped conv_transpose1d weight shape: (in_channels, out_channels_per_group, kernel_size)
    weight_grouped = torch.randn(in_channels, out_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    # Our custom kernel expects weight of shape (in_channels, out_channels, kernel_size)
    # So we deliberately pass a weight tensor unsuited for groups.
    # Also, native conv_transpose1d will be called with groups=2.
    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)

    # Run custom kernel forward using the grouped weight (which does NOT have the proper shape).
    custom_out = cuda_module.forward(x, weight_grouped, None, stride, padding, dilation)
    # Run native PyTorch conv_transpose1d with groups=2.
    native_out = torch.nn.functional.conv_transpose1d(x, weight_grouped, None, stride, padding, dilation, groups=groups)

    # The output channels will be different:
    # custom_out.shape[1] == weight_grouped.shape[1] (i.e. out_channels_per_group)
    # native_out.shape[1] == groups * (out_channels_per_group)
    assert custom_out.shape[1] != native_out.shape[1], "Custom kernel incorrectly handles grouped convolution."

if __name__ == "__main__":
    pytest.main([os.path.realpath(__file__)])
