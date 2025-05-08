
import torch
import torch.nn.functional as F
import pytest
from torch.utils.cpp_extension import load
import os

# Helper: Build the CUDA extension from kernel.cu
def build_kernel():
    # Ensure the path to kernel.cu is correct.
    src_path = os.path.join(os.path.dirname(__file__), "kernel.cu")
    cuda_module = load(
        name="test_conv1d",
        sources=[src_path],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 & 2: Test when L_out is not a multiple of 4, forcing non-vectorized store and checking work distribution.
def test_non_multiple_of_four_output():
    # Set up parameters: using kernel_size=3 with no padding and stride=1 yields L_out = L_in - 2.
    # Choose L_in=7 gives L_out=5 (not a multiple of 4).
    batch_size = 2
    in_channels = 4
    out_channels = 8
    kernel_size = 3
    L_in = 7
    stride = 1
    padding = 0
    dilation = 1
    groups = 1

    # Create input tensor
    x = torch.randn(batch_size, in_channels, L_in, device="cuda", dtype=torch.float32)
    # Weight shape: [C_out, C_in/groups, K]
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    bias = None  # Not using bias

    # Expected result using PyTorch function
    expected = F.conv1d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # Call custom CUDA kernel through the extension
    conv_module = build_kernel()
    # Our extension's forward has the signature: forward(x, weight, bias (or None), stride, padding, dilation, groups)
    result = conv_module.forward(x, weight, None, stride, padding, dilation, groups)

    torch.cuda.synchronize()
    # Use a relatively loose tolerance because the fallback scalar store may have slight differences.
    assert torch.allclose(result, expected, atol=1e-5), \
        f"Issue with non-multiple-of-4 output: max diff {(result-expected).abs().max()}"

# Issue 3: Test with a number of input channels that is not divisible by groups.
def test_invalid_group_division():
    # Set parameters where in_channels is not divisible by groups.
    batch_size = 2
    in_channels = 3  # not divisible by groups=2
    out_channels = 4
    kernel_size = 3
    L_in = 10
    stride = 1
    padding = 1
    dilation = 1
    groups = 2

    x = torch.randn(batch_size, in_channels, L_in, device="cuda", dtype=torch.float32)
    # Normally PyTorch conv1d requires in_channels % groups == 0;
    # We'll bypass PyTorch's check by constructing weight with shape based on integer division.
    group_size_in = in_channels // groups  # This rounds down (3//2 == 1)
    weight = torch.randn(out_channels, group_size_in, kernel_size, device="cuda", dtype=torch.float32)
    bias = None

    # Compute expected output using PyTorch's conv1d cannot be done directly since dimensions are incompatible.
    # Instead, we run the custom kernel and expect the result to be different from a "correct" convolution.
    conv_module = build_kernel()
    result = conv_module.forward(x, weight, None, stride, padding, dilation, groups)
    torch.cuda.synchronize()

    # Since the grouping assumption is violated, we expect the result to be incorrect.
    # Here we simply check that result does not match a convolution computed by repeating the weight for missing channels.
    # We construct a "patched" weight by repeating the available weight channels.
    # For testing purposes, we repeat along the missing channel dimension.
    weight_patched = weight.repeat(1, groups, 1)
    expected = F.conv1d(x, weight_patched, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    # We expect the custom kernel result to differ.
    assert not torch.allclose(result, expected, atol=1e-5), \
        "Kernel did not exhibit error behavior when C_in is not divisible by groups."

# Issue 4: Test with a kernel size that is not a small constant (forcing the loop not to unroll at compile time)
def test_non_compile_time_kernel_size():
    batch_size = 2
    in_channels = 4
    out_channels = 8
    # Use a larger kernel size that is less likely to be unrolled fully at compile time.
    kernel_size = 5
    L_in = 12
    stride = 1
    padding = 2  # typical to keep output length same as input length
    dilation = 1
    groups = 1

    x = torch.randn(batch_size, in_channels, L_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    bias = None

    expected = F.conv1d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    conv_module = build_kernel()
    result = conv_module.forward(x, weight, None, stride, padding, dilation, groups)
    torch.cuda.synchronize()

    assert torch.allclose(result, expected, atol=1e-5), \
        f"Issue with non-compile-time kernel size: max diff {(result-expected).abs().max()}"

if __name__ == "__main__":
    pytest.main([__file__])
