
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA kernel from kernel.cu
def build_kernel():
    # Assuming the file kernel.cu is in the current directory.
    cuda_module = load(
        name="depthwise_conv_kernel",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Utility routine to compute reference convolution output using torch.nn.functional.conv2d
def reference_depthwise_conv2d(x, weight, bias, stride, padding, dilation, groups):
    # Note: For depthwise convolution in PyTorch, weight shape is (out_channels, 1, kH, kW)
    return torch.nn.functional.conv2d(x, weight, bias=bias, stride=stride, padding=padding,
                                        dilation=dilation, groups=groups)

# Test 1: Incorrect input-channel indexing in grouped convolution.
# In this test we deliberately choose groups such that channels_per_group > 1.
def test_incorrect_input_indexing():
    device = "cuda"
    # Setup a scenario where in_channels = groups * channels_per_group with channels_per_group > 1.
    batch_size = 1
    groups = 2
    channels_per_group = 2
    in_channels = groups * channels_per_group  # 4 channels total.
    height, width = 16, 16
    kernel_h, kernel_w = 3, 3
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    bias_flag = False

    # Create input tensor and weight tensor
    x = torch.randn(batch_size, in_channels, height, width, device=device, dtype=torch.float32)
    # For a grouped convolution with channels_per_group > 1, the weight shape is (out_channels, 1, kh, kw)
    # where out_channels == in_channels. We simulate that by constructing weight with shape (groups * channels_per_group, 1, kh, kw)
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device=device, dtype=torch.float32)
    bias_tensor = None

    # Get reference output using PyTorch's conv2d
    ref_output = torch.nn.functional.conv2d(x, weight, bias=bias_tensor,
                                              stride=stride, padding=padding,
                                              dilation=dilation, groups=groups)

    # Load our custom CUDA kernel module.
    kernel_module = build_kernel()
    # Our kernel expects weight of shape (groups, channels_per_group, kernel_h, kernel_w).
    # Reshape accordingly.
    weight_reshaped = weight.view(groups, channels_per_group, kernel_h, kernel_w)

    # Invoke our kernel. It uses the following arguments:
    # x, weight, bias, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups.
    custom_output = kernel_module.forward(
        x,
        weight_reshaped,
        None,  # no bias
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        groups
    )

    # Because of the incorrect input indexing in the kernel, the output will differ from the reference.
    # We assert that they are not close.
    assert not torch.allclose(custom_output, ref_output, atol=1e-4), \
        "Custom kernel output unexpectedly matches the reference output despite incorrect indexing."

# Test 2: Incorrect behavior when input data type is not float32.
def test_non_float_input():
    device = "cuda"
    batch_size = 1
    in_channels = 3
    height, width = 16, 16
    kernel_h, kernel_w = 3, 3
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    groups = in_channels
    bias_flag = False

    # Create double precision (float64) input tensor
    x = torch.randn(batch_size, in_channels, height, width, device=device, dtype=torch.float64)
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device=device, dtype=torch.float64)
    # Reshape weight to expected shape (groups, channels_per_group, kernel_h, kernel_w).
    weight_reshaped = weight.view(groups, 1, kernel_h, kernel_w)

    kernel_module = build_kernel()

    # The kernel does not support double precision and using data_ptr<float>() on a double tensor
    # should lead to an error or undefined behavior. We check that an error is raised.
    with pytest.raises(Exception):
        _ = kernel_module.forward(
            x,
            weight_reshaped,
            None,
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
            groups
        )

# Test 3: Ensure that non-CUDA tensors are rejected.
def test_non_cuda_input():
    device = "cpu"  # Intentionally use CPU tensor.
    batch_size = 1
    in_channels = 3
    height, width = 16, 16
    kernel_h, kernel_w = 3, 3
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    groups = in_channels
    bias_flag = False

    x = torch.randn(batch_size, in_channels, height, width, device=device, dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device=device, dtype=torch.float32)
    # For the kernel, we need to reshape the weight and then move it to cuda.
    weight_reshaped = weight.view(groups, 1, kernel_h, kernel_w)
    kernel_module = build_kernel()

    with pytest.raises(RuntimeError):
        _ = kernel_module.forward(
            x,
            weight_reshaped,
            None,
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
            groups
        )
