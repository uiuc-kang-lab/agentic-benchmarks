
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility to build the CUDA extension from kernel.cu.
def build_kernel():
    return load(
        name="depthwise_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

# Test 1: Trigger the non-float32 data type issue.
#
# This test creates double-precision input, weight and (if used) bias.
# Since the kernel internally casts to float* and does not check data type,
# it is expected to produce wrong results or crash.
def test_non_float32_dtype():
    mod = build_kernel()
    batch_size = 2
    in_channels = 3
    kernel_size = 3
    stride = 1
    padding = 0

    # Create input and weight as double precision
    input = torch.randn(batch_size, in_channels, 16, 16, dtype=torch.float64, device='cuda')
    # For depthwise convolution groups=in_channels, channel_per_group==1,
    # so weight shape should be (in_channels, 1, kernel_size, kernel_size)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, dtype=torch.float64, device='cuda')
    bias = torch.randn(in_channels, dtype=torch.float64, device='cuda')

    with pytest.raises(RuntimeError):
        # Expect that a runtime error occurs due to reinterpretation of double data as float.
        mod.forward(input, weight, bias, stride, padding)

# Test 2: Trigger the grouped convolution issue.
#
# This test creates a grouped convolution scenario where channels_per_group != 1.
# The kernel is written to assume depthwise (channels_per_group==1). When running
# a grouped convolution where each group has multiple channels, the weight indexing
# will be wrong and the output will not match PyTorch's built-in grouped convolution.
def test_grouped_convolution_mismatch():
    mod = build_kernel()
    batch_size = 1
    in_channels = 4
    groups = 2  # groups != in_channels => channels_per_group > 1 is required.
    channels_per_group = in_channels // groups  # Here, 4/2 = 2.
    kernel_size = 3
    stride = 1
    padding = 1

    # Create input weight and bias with proper dimensions for grouped convolution:
    # Input: (batch, in_channels, H, W)
    # Weight: (in_channels, channels_per_group, kernel_size, kernel_size)
    # Bias: (in_channels * channels_per_group,)
    H, W = 8, 8
    input = torch.randn(batch_size, in_channels, H, W, dtype=torch.float32, device='cuda')
    weight = torch.randn(in_channels, channels_per_group, kernel_size, kernel_size,
                          dtype=torch.float32, device='cuda')
    # The expected output channels for a grouped convolution is in_channels * channels_per_group.
    bias = torch.randn(in_channels * channels_per_group, dtype=torch.float32, device='cuda')

    # Run the custom kernel.
    output_custom = mod.forward(input, weight, bias, stride, padding)
    # For reference, use PyTorch's built-in grouped convolution.
    ref_conv = torch.nn.Conv2d(
        in_channels,
        in_channels * channels_per_group,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=in_channels,  # This matches depthwise conv; our weight here is not for depthwise.
        bias=True
    ).to('cuda')

    # Manually assign our weight and bias to ref_conv.
    # For the built-in function to have the same weights, we need to reshape our weight.
    # The weight expected by nn.Conv2d, when groups==in_channels, is of shape
    # (in_channels, 1, kernel_size, kernel_size). Here our weight has shape (in_channels, channels_per_group, ...),
    # so the custom kernel and the PyTorch function use different conventions.
    # Hence, we expect a mismatch.
    ref_conv.weight.data.copy_(weight.view(in_channels, 1, kernel_size, kernel_size).repeat(1, channels_per_group, 1, 1))
    ref_conv.bias.data.copy_(bias)

    output_ref = ref_conv(input)
    # The outputs should differ due to the incorrect handling of grouped convolution.
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(output_custom, output_ref, atol=1e-5)

# Test 3: Trigger grid dimension (gridDim.z) limitation.
#
# This test creates an input that will require a grid.z value exceeding the maximum.
# CUDA typically limits gridDim.z to 65535. Here, we synthesize an input
# whose batch_size*out_channels exceeds that limit which should cause a kernel launch error.
def test_grid_dimension_limit():
    mod = build_kernel()
    # Set parameters such that batch_size * out_channels > 65535.
    # For depthwise convolution, out_channels equals in_channels.
    in_channels = 1
    batch_size = 70000  # 70000 > 65535, so gridDim.z = 70000 should be illegal.
    kernel_size = 3
    stride = 1
    padding = 1

    H, W = 8, 8
    input = torch.randn(batch_size, in_channels, H, W, dtype=torch.float32, device='cuda')
    # For depthwise conv, weight shape is (in_channels, 1, kernel_size, kernel_size)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, dtype=torch.float32, device='cuda')
    bias = torch.randn(in_channels, dtype=torch.float32, device='cuda')

    with pytest.raises(RuntimeError):
        # Expect a runtime error due to an illegal grid dimension.
        mod.forward(input, weight, bias, stride, padding)
