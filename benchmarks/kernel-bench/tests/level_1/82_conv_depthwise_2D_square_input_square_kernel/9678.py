
import torch
import torch.nn as nn
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

def custom_depthwise_conv2d(x, weight, bias, stride, padding, groups, module):
    # Call the CUDA kernel wrapped function.
    return module.forward(x, weight, bias, stride, padding, groups)

# Issue 1: groups mismatch
# If groups != in_channels, the kernel performs wrong indexing.
def test_groups_mismatch(kernel_module):
    batch_size = 1
    in_channels = 3
    height = 32
    width = 32
    kernel_size = 3
    stride = 1
    padding = 1

    # Create a depthwise convolution model that expects groups == in_channels.
    conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=True)
    conv = conv.cuda()

    x = torch.randn(batch_size, in_channels, height, width, device="cuda")
    # Create weight with PyTorch layout: (in_channels, 1, kernel_size, kernel_size)
    weight = conv.weight.clone()
    bias = conv.bias.clone()

    # Now call the kernel with an incorrect groups parameter (e.g. groups=1)
    kernel_out = custom_depthwise_conv2d(x, weight, bias, stride, padding, groups=1, module=kernel_module)

    # The standard conv2d (which obeys groups) produces a different result.
    expected = conv(x)
    # We expect that the kernel’s output does not match the reference due to the groups mismatch.
    assert not torch.allclose(kernel_out, expected, atol=1e-5), (
        f"Kernel output unexpectedly matches reference output even though groups mismatch."
    )

# Issue 2: non-square input/kernel
# Using non-square kernels (or inputs) will break the assumptions on kernel dimensions.
def test_non_square_kernel(kernel_module):
    batch_size = 1
    in_channels = 3
    height = 32
    width = 32
    # Create a non-square kernel (e.g. 3x5 instead of 3x3)
    kernel_h = 3
    kernel_w = 5
    stride = 1
    padding = 1

    x = torch.randn(batch_size, in_channels, height, width, device="cuda")
    # Force a non-square weight tensor. Note that the kernel implementation uses weight.size(2)
    # for both height and width indices. This should trigger an out-of-bound access or wrong results.
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device="cuda")
    bias = torch.randn(in_channels, device="cuda")

    with pytest.raises(RuntimeError):
        # Expect the kernel to complain (e.g. via an illegal memory access) because
        # the inner loop uses kernel_w wrongly.
        _ = custom_depthwise_conv2d(x, weight, bias, stride, padding, groups=in_channels, module=kernel_module)

# Issue 3: non-contiguous inputs
# The kernel assumes contiguous tensors. Non-contiguous inputs may lead to wrong results.
def test_non_contiguous_input(kernel_module):
    batch_size = 1
    in_channels = 3
    height = 32
    width = 32
    kernel_size = 3
    stride = 1
    padding = 1

    # Standard depthwise conv2d (with groups==in_channels) as reference.
    conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=True)
    conv = conv.cuda()

    # Create a contiguous input and weight.
    x = torch.randn(batch_size, in_channels, height, width, device="cuda")
    weight = conv.weight.clone()
    bias = conv.bias.clone()

    # Make the input non-contiguous by transposing and then transposing back (without calling contiguous())
    x_noncontig = x.permute(0, 2, 3, 1)  # Now shape is (batch_size, height, width, in_channels)
    x_noncontig = x_noncontig.permute(0, 3, 1, 2)  # Shape restored but tensor is not contiguous
    
    kernel_out = custom_depthwise_conv2d(x_noncontig, weight, bias, stride, padding, groups=in_channels, module=kernel_module)
    expected = conv(x)
    # The output should differ because the kernel does not handle non-contiguous inputs.
    assert not torch.allclose(kernel_out, expected, atol=1e-5), (
        "Kernel output matches reference with non-contiguous input, but it should not."
    )
