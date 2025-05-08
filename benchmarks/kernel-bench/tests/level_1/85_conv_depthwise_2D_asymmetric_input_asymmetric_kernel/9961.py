
import torch
import pytest
from torch.utils.cpp_extension import load
import torch.nn.functional as F

def build_kernel():
    cuda_module = load(
        name="custom_depthwise_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test case 1: Passing a tensor with dtype float64 should trigger an issue (only float32 is supported)
def test_dtype_issue():
    module = build_kernel()
    batch_size = 2
    in_channels = 3
    height = 16
    width = 16
    kernel_h = 3
    kernel_w = 5

    # Create input and weight with float64, which is not supported by the kernel
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float64, device="cuda")
    # For a depthwise convolution, weight is expected to be of shape (in_channels, 1, kH, kW)
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, dtype=torch.float64, device="cuda")
    bias = torch.randn(in_channels, dtype=torch.float64, device="cuda")

    # The forward function in the CUDA extension does not perform type checking.
    # It assumes float pointers. This should cause the kernel to produce
    # unexpected results when using double precision.
    output = module.forward(
        x, weight, bias,
        stride_h=1, stride_w=1,
        padding_h=0, padding_w=0,
        dilation_h=1, dilation_w=1,
        groups=in_channels
    ).to(dtype=torch.float64)
    # Use PyTorch's built-in conv2d for correct result reference (it supports double)
    ref = F.conv2d(
        x, weight, bias,
        stride=(1, 1), padding=(0, 0),
        dilation=(1, 1), groups=in_channels
    )
    # The results should not match because our kernel incorrectly interprets the bits
    # when provided with double (and may also silently fail).
    assert not torch.allclose(output, ref, atol=1e-5), \
        "Test failed: The kernel incorrectly accepted float64 input and produced a valid result."

# Test case 2: Using a grouped convolution where channels per group > 1 should trigger an issue.
def test_grouped_conv_issue():
    module = build_kernel()
    batch_size = 2
    # Use a grouped convolution where each group has more than one input channel.
    in_channels = 4
    groups = 2  # in_channels/groups = 2 (i.e. channels per group > 1)
    # For general grouped convolution, the expected weight shape is (out_channels, in_channels/groups, kH, kW).
    # However, the kernel is written as if weight shape is (groups, channels_per_group, kH, kW) and sums only one input channel.
    # To match the kernel expectations, we will supply a weight tensor with shape (groups, in_channels//groups, kH, kW)
    kernel_h = 3
    kernel_w = 3

    # Generate random input tensor.
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    # Create weight tensor with shape (groups, channels_per_group, kH, kW)
    weight = torch.randn(groups, in_channels // groups, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    # Bias of shape (out_channels) where out_channels = groups * (in_channels//groups)
    bias = torch.randn(groups * (in_channels // groups), device="cuda", dtype=torch.float32)

    # Run the custom CUDA kernel. Note that the kernel forward() computes:
    #   out_channels = groups * weight.size(1)
    # and then uses a single accumulation over the input channel block.
    output = module.forward(
        x, weight, bias,
        stride_h=1, stride_w=1,
        padding_h=1, padding_w=1,
        dilation_h=1, dilation_w=1,
        groups=groups
    )
    
    # Compute the reference output using PyTorch's conv2d for grouped convolution.
    # PyTorch expects weight in shape (out_channels, in_channels/groups, kH, kW)
    weight_reshaped = weight.view(groups * (in_channels // groups), 1, kernel_h, kernel_w)
    ref = F.conv2d(
        x, weight_reshaped, bias,
        stride=(1,1), padding=(1,1),
        dilation=(1,1), groups=groups
    )
    
    # Because the custom kernel does not sum over all input channels in each group,
    # the output should differ from the reference.
    assert not torch.allclose(output, ref, atol=1e-5), \
        "Test failed: The kernel produced correct results for grouped convolution, but an issue was expected."

if __name__ == "__main__":
    pytest.main([__file__])
