
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# Helper to compile and load our CUDA extension
def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Incorrect indexing for weights in grouped convolution.
# This test creates a small input with groups > 1,
# uses the custom CUDA kernel and compares it with PyTorch's nn.ConvTranspose2d.
# The outputs should differ due to the incorrect weight indexing.
def test_incorrect_weight_indexing():
    torch.manual_seed(0)
    batch_size = 2
    in_channels = 4
    out_channels = 8   # must be divisible by groups
    groups = 2
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    height, width = 5, 5

    # Define a reference module using nn.ConvTranspose2d
    ref_conv = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding,
        dilation=dilation, groups=groups, bias=True
    ).cuda()

    # Grab the weights and bias from the PyTorch module:
    weight = ref_conv.weight.detach().clone()
    bias = ref_conv.bias.detach().clone()

    # Define an input tensor:
    x = torch.randn(batch_size, in_channels, height, width, device='cuda')

    # Run the reference convolution
    ref_out = ref_conv(x)

    # Run the custom CUDA kernel forward procedure:
    # Note: The custom kernel expects weight of shape (in_channels, out_channels_per_group, kernel_h, kernel_w)
    # and bias of shape (out_channels_per_group * groups). They are exactly as defined in PyTorch's ConvTranspose2d.
    custom_module = build_kernel()
    out_custom = custom_module.forward(
        x, weight, bias, 
        [stride[0], stride[1]],
        [padding[0], padding[1]],
        [dilation[0], dilation[1]],
        groups
    )

    # They should produce the same result if the kernel were correct.
    # Here we expect a discrepancy due to the weight indexing bug.
    assert not torch.allclose(ref_out, out_custom, atol=1e-5), "The custom kernel produced the same output as reference even though the weight indexing bug should cause differences."

# Issue 2: Kernel type support is limited to float32.
# Here we deliberately pass double precision tensors to trigger a potential failure or incorrect results.
def test_incorrect_dtype():
    batch_size = 2
    in_channels = 4
    out_channels = 4
    groups = 1
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    height, width = 5, 5

    # Create input, weight, and bias with dtype float64
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float64)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], device='cuda', dtype=torch.float64)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float64)

    custom_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expecting an error or failure because the kernel uses float atomicAdd which is not defined for double.
        _ = custom_module.forward(
            x, weight, bias,
            [stride[0], stride[1]],
            [padding[0], padding[1]],
            [dilation[0], dilation[1]],
            groups
        )

# Issue 3: The initialization kernel does not check for launch errors.
# We trigger an error in the bias initialization phase by providing a bias tensor with an incorrect shape.
def test_bias_shape_error():
    batch_size = 2
    in_channels = 4
    out_channels = 4
    groups = 1
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    height, width = 5, 5

    # Create proper input and weight but an incorrect bias (wrong shape)
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], device='cuda', dtype=torch.float32)
    # Provide a bias with an incorrect number of elements.
    bias = torch.randn(out_channels - 1, device='cuda', dtype=torch.float32)

    custom_module = build_kernel()
    # The custom kernel does not check bias boundaries so this may lead to an illegal memory access.
    with pytest.raises(Exception):
        _ = custom_module.forward(
            x, weight, bias,
            [stride[0], stride[1]],
            [padding[0], padding[1]],
            [dilation[0], dilation[1]],
            groups
        )

if __name__ == '__main__':
    pytest.main([__file__])
