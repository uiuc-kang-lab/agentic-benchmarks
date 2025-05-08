
import torch
import pytest
from torch.utils.cpp_extension import load
import torch.nn.functional as F

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function: performs convolution using the custom kernel via the extension.
def custom_conv2d(x, weight, bias, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups):
    mod = build_kernel()
    return mod.forward(x, weight, bias, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups)

# Test 1: Non-float32 data type to trigger dtype issue.
def test_non_float32_dtype():
    # We create an input tensor with float64.
    batch_size = 2
    in_channels = 3
    height, width = 16, 16
    kernel_size = 3
    
    # Create float64 tensors.
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float64, device="cuda")
    # For a depthwise conv, weight shape is (in_channels, 1, kernel_size, kernel_size)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, dtype=torch.float64, device="cuda")
    bias = torch.randn(in_channels, dtype=torch.float64, device="cuda")
    
    stride = 1
    padding = 1
    dilation = 1
    groups = in_channels

    with pytest.raises(RuntimeError):
        # The kernel is written for float32; using float64 should trigger an error or misbehavior.
        custom_conv2d(x, weight, bias, stride, stride, padding, padding, dilation, dilation, groups)

# Test 2: General grouped convolution (groups != in_channels, channels_per_group > 1)
def test_invalid_grouped_convolution():
    # The kernel assumes a depthwise conv (each group has one input channel).
    # We create a case where groups != in_channels.
    batch_size = 2
    in_channels = 4   # Total input channels.
    groups = 2        # Expecting two channels per group.
    # For general grouped convolution, the number of channels per group is in_channels / groups.
    channels_per_group = in_channels // groups  # 2 channels per group.
    out_channels_per_group = 1  # typical depthwise conv uses 1, but here we force a scenario.
    out_channels = groups * out_channels_per_group
    height, width = 16, 16
    kernel_h, kernel_w = 3, 3

    # Create float32 tensors.
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device="cuda")
    # For general grouped convolution, PyTorch weight shape is (groups, out_channels_per_group, kernel_h, kernel_w)
    weight = torch.randn(groups, out_channels_per_group, kernel_h, kernel_w, dtype=torch.float32, device="cuda")
    # PyTorch Conv2d expects weight to be reshaped to (groups * out_channels_per_group, 1, kernel_h, kernel_w)
    # but our kernel uses a flat weight assuming depthwise style.
    # We simulate a “grouped” conv by not expanding the input channels.
    # In a correct implementation, the convolution would sum over channels in the same group.
    bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")
    
    # Compute PyTorch reference result using grouped convolution.
    conv = torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=(kernel_h, kernel_w),
        stride=1, padding=1, dilation=1, groups=groups, bias=True
    ).to("cuda")
    # Set the conv weights and bias to our created ones.
    # For a general grouped convolution the weight shape is (out_channels, in_channels/groups, kernel_h, kernel_w).
    # We fill conv.weight with our weight repeated appropriately so that conv.weight[:,0,:,:] equals weight for each group.
    with torch.no_grad():
        for g in range(groups):
            for oc in range(out_channels_per_group):
                # Our kernel expects one filter per output channel.
                conv.weight.data[g*out_channels_per_group+oc].copy_( weight[g, oc].expand(channels_per_group, kernel_h, kernel_w) )
        conv.bias.data.copy_(bias)
    
    ref = conv(x)

    # Run our custom kernel.
    # Our custom kernel is designed for depthwise conv (each output channel reads one input channel)
    # so when groups != in_channels the output will be computed incorrectly.
    out_custom = custom_conv2d(x, weight.view(out_channels, 1, kernel_h, kernel_w), bias,
                               1, 1, 1, 1, 1, 1, groups)
    
    # We expect the output computed by our custom kernel to differ significantly from the reference.
    diff = (ref - out_custom).abs().max().item()
    assert diff > 1e-3, f"Expected large discrepancy due to invalid grouped convolution handling, but got max difference {diff}"
