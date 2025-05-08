
import torch
import torch.nn.functional as F
import pytest
from torch.utils.cpp_extension import load
import os

# Assume that kernel.cu is in the same directory as this test file.
# Build the extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test Case 1: Trigger issue with groups != in_channels
# This test creates a grouped convolution where each group contains more than one channel.
# The custom kernel (which incorrectly indexes input) should yield a different result from the reference.
def test_grouped_convolution_wrong_indexing():
    # Setup parameters such that groups != in_channels; for example:
    # in_channels = 4, groups = 2, so channels per group = 4/2 = 2.
    batch_size = 2
    in_channels = 4
    groups = 2
    channels_per_group = in_channels // groups
    kernel_h, kernel_w = 3, 3
    height, width = 16, 16
    stride = 1
    padding = 1
    dilation = 1

    # Create input tensor and weight tensor that match the grouped convolution format.
    # For a grouped convolution, weight shape is (out_channels, in_channels/groups, kernel_h, kernel_w)
    # Here out_channels == groups * channels_per_group = in_channels.
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, channels_per_group, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    # Compute the reference output using PyTorch's F.conv2d
    ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # Call the custom CUDA kernel via the extension.
    my_module = build_kernel()
    output = my_module.forward(
        x, weight, bias, 
        stride, stride, 
        padding, padding,
        dilation, dilation,
        groups
    )
    torch.cuda.synchronize()

    # Because of the wrong input indexing, the outputs should differ.
    # We check that they are not close.
    assert not torch.allclose(output, ref, atol=1e-5), \
        "The custom kernel produced the same output as the reference; expected an error due to wrong input indexing."

# Test Case 2: Trigger an issue when input tensor type is not float32
# The kernel is hard-coded with float; using float16 should lead to errors or incorrect computation.
def test_incorrect_dtype():
    batch_size = 2
    in_channels = 3  # Depthwise convolution typically expects groups equal to in_channels.
    groups = in_channels
    kernel_h, kernel_w = 3, 5
    height, width = 32, 32
    stride = 1
    padding = 0
    dilation = 1

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float16)
    # Create weight with shape: (in_channels, 1, kernel_h, kernel_w) as used in depthwise conv.
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float16)
    bias = None

    my_module = build_kernel()
    # The kernel expects float32 inputs. Depending on the device and extension,
    # either an error is raised or the computation is wrong.
    with pytest.raises(Exception):
        # We expect an exception due to the wrong dtype.
        _ = my_module.forward(
            x, weight, bias,
            stride, stride,
            padding, padding,
            dilation, dilation,
            groups
        )
