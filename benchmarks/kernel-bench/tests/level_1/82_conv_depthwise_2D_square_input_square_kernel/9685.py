
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Utility to build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="depthwise_conv_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Groups parameter misuse.
# This test creates a dummy convolution scenario where groups != in_channels.
# The expected behavior is that our kernel (which ignores groups) produces a result
# different from PyTorch's implementation when groups is not equal to in_channels.
def test_groups_parameter_mismatch():
    cuda_module = build_kernel()

    batch_size = 4
    in_channels = 4
    kernel_size = 3
    height = width = 16
    stride = 1
    padding = 1
    # Here, groups is set to 2 which is not equal to in_channels.
    groups = 2

    # Create an input tensor.
    x = torch.randn(batch_size, in_channels, height, width, device="cuda")
    # Create a weight tensor with shape: (in_channels, 1, kernel_size, kernel_size)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda")
    bias = torch.randn(in_channels, device="cuda")

    # Run the custom CUDA kernel.
    out_custom = cuda_module.forward(x, weight, bias, stride, padding, groups)

    # Use PyTorch's own conv2d (which supports groups properly) for reference.
    conv = torch.nn.Conv2d(
        in_channels, in_channels, kernel_size,
        stride=stride, padding=padding, groups=groups, bias=True
    ).to("cuda")
    # Manually set conv weight and bias from our tensors for consistency.
    # For depthwise conv, PyTorch expects the weight to have shape (in_channels, 1, k, k)
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)
    out_ref = conv(x)
    
    # Expect that the custom kernel, which ignores the provided groups parameter, produces
    # an output that does not match the PyTorch conv2d reference.
    # Using a loose tolerance because differences may be nonzero.
    assert not torch.allclose(out_custom, out_ref, atol=1e-4), (
        "Custom kernel output should differ when groups != in_channels, but it did not."
    )

# Test 2: Data type support.
# Use a half-precision tensor. Since our kernel only handles float and double,
# we expect the extension dispatch to throw an error.
def test_half_precision_input():
    cuda_module = build_kernel()

    batch_size = 2
    in_channels = 3
    kernel_size = 3
    height = width = 8
    stride = 1
    padding = 1
    groups = in_channels  # valid depthwise configuration

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float16)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float16)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError):
        # Expect a runtime error due to unsupported dtype.
        _ = cuda_module.forward(x, weight, bias, stride, padding, groups)

# Test 3: Non-contiguous input tensor.
# The kernel assumes a particular memory layout and indexing arithmetic.
# If the input tensor is non-contiguous, the computed indices may be incorrect.
def test_non_contiguous_input():
    cuda_module = build_kernel()

    batch_size = 4
    in_channels = 3
    kernel_size = 3
    height = width = 16
    stride = 1
    padding = 1
    groups = in_channels

    # Create a contiguous tensor and then create a non-contiguous view by transposing.
    x = torch.randn(batch_size, in_channels, height, width, device="cuda")
    # Transpose h and w to make it non-contiguous, then transpose back to original order.
    x_noncontig = x.transpose(2, 3).transpose(2, 3)  # still same shape but non-contiguous
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda")
    bias = torch.randn(in_channels, device="cuda")

    # Check that x_noncontig is indeed non-contiguous.
    assert not x_noncontig.is_contiguous(), "x_noncontig should be non-contiguous"

    # Run the custom kernel.
    out_custom = cuda_module.forward(x_noncontig, weight, bias, stride, padding, groups)

    # For reference, run PyTorch's convolution on the contiguous input.
    conv = torch.nn.Conv2d(
        in_channels, in_channels, kernel_size,
        stride=stride, padding=padding, groups=groups, bias=True
    ).to("cuda")
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)
    out_ref = conv(x_noncontig)

    # Due to non-contiguity, the kernel may compute incorrect indices.
    # We expect a mismatch between the custom kernel and PyTorch's own conv2d.
    assert not torch.allclose(out_custom, out_ref, atol=1e-4), (
        "Custom kernel output should differ for non-contiguous input, indicating an indexing issue."
    )

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
