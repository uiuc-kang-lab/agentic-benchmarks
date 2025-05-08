
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Ensure that the kernel.cu file exists in the same directory as this test file.
KERNEL_FILE = os.path.join(os.path.dirname(__file__), "kernel.cu")


def build_kernel():
    cuda_module = load(
        name="transposed_conv3d_module",
        sources=[KERNEL_FILE],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper: Compare with PyTorch's built-in ConvTranspose3d with no bias
def reference_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    conv = torch.nn.ConvTranspose3d(
        in_channels=x.size(1),
        out_channels=weight.size(1) * groups,
        kernel_size=weight.shape[2:5],
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=(bias is not None)
    )
    # Set conv weight and bias to match the provided tensors.
    # PyTorch expects the weight shape for ConvTranspose3d to be (in_channels, out_channels/groups, kT, kH, kW)
    conv.weight.data.copy_(weight)
    if bias is not None:
        conv.bias.data.copy_(bias)
    return conv(x)

# Issue 1: Kernel supports only float32 and double. Using half precision should trigger an error.
def test_dtype_support():
    cuda_module = build_kernel()
    # Create half precision input, weight and (optionally) bias
    batch_size = 2
    in_channels = 4
    out_channels = 8
    kernel_size = (3, 3, 3)
    depth, height, width = 5, 6, 7
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    output_padding = (1, 1, 1)
    groups = 1

    x = torch.randn(batch_size, in_channels, depth, height, width, dtype=torch.half, device='cuda')
    weight = torch.randn(in_channels, out_channels // groups, *kernel_size, dtype=torch.half, device='cuda')
    bias = None

    with pytest.raises(RuntimeError):
        # This call should fail because half precision is not handled by AT_DISPATCH_FLOATING_TYPES.
        _ = cuda_module.forward(x, weight, bias, list(stride), list(padding), list(output_padding), groups)

# Issue 2: Group handling does not check for even divisibility.
def test_group_divisibility():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 4
    # Intentionally set out_channels such that out_channels/groups is not an integer.
    out_channels = 7  # 7 cannot be divided evenly by groups=2.
    kernel_size = (3, 3, 3)
    depth, height, width = 5, 6, 7
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    output_padding = (0, 0, 0)
    groups = 2

    x = torch.randn(batch_size, in_channels, depth, height, width, device='cuda', dtype=torch.float32)
    # Based on conv_transpose weight shape convention:
    # Weight shape should be (in_channels, out_channels/groups, kT, kH, kW)
    # Here, out_channels/groups is 7/2 which is non-integer. We force a weight shape that is erroneous.
    weight = torch.randn(in_channels, (out_channels + 1) // groups, *kernel_size, device='cuda', dtype=torch.float32)
    bias = None

    # In a well‐checked implementation, an error should be raised.
    # Here we check that the kernel produces an output that does not match the PyTorch reference.
    output = cuda_module.forward(x, weight, bias, list(stride), list(padding), list(output_padding), groups)
    output_ref = reference_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups)
    # They likely differ; so we assert that they are not equal.
    assert not torch.allclose(output, output_ref, atol=1e-5), "Kernel unexpectedly produced correct result with uneven group division."

# Issue 3: The modulo operations on negative indices might lead to unexpected results.
def test_negative_padding_behavior():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 8
    kernel_size = (4, 4, 4)  # Larger kernel
    depth, height, width = 5, 5, 5
    stride = (2, 2, 2)
    # Set padding smaller than what might be needed given the kernel size, to force negative intermediate indices
    padding = (0, 0, 0)
    output_padding = (0, 0, 0)
    groups = 1

    x = torch.randn(batch_size, in_channels, depth, height, width, device='cuda', dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels // groups, *kernel_size, device='cuda', dtype=torch.float32)
    bias = None

    output = cuda_module.forward(x, weight, bias, list(stride), list(padding), list(output_padding), groups)
    output_ref = reference_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups)
    # Because of the negative indexing issue, the outputs may differ.
    assert not torch.allclose(output, output_ref, atol=1e-5), "Kernel output unexpectedly matches reference despite potential negative index issues."

# Issue 4: Lack of error checking after kernel launch. Although we cannot directly catch the missing error-check,
# we can simulate an erroneous launch configuration by providing wrong output dimensions.
def test_kernel_launch_error_propagation():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 8
    kernel_size = (3, 3, 3)
    depth, height, width = 5, 6, 7
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    output_padding = (0, 0, 0)
    groups = 1

    x = torch.randn(batch_size, in_channels, depth, height, width, device='cuda', dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels // groups, *kernel_size, device='cuda', dtype=torch.float32)
    bias = None

    # Corrupt the weight tensor shape by providing an extra element (simulate misconfiguration)
    weight_bad = torch.randn(in_channels, out_channels // groups + 1, *kernel_size, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(x, weight_bad, bias, list(stride), list(padding), list(output_padding), groups)
