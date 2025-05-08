
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension (kernel.cu should be in the same directory)
def build_kernel():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_file = os.path.join(this_dir, "kernel.cu")
    cuda_module = load(
        name="depthwise_conv_cuda",
        sources=[cuda_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test Case 1: Grid Dimension Issue
# This test attempts to trigger a failure by forcing a grid dimension (z dimension)
# that exceeds typical CUDA limits (e.g., > 65,535). It sets a huge batch size.
def test_grid_dimension_overflow():
    cuda_module = build_kernel()
    # Create an input with a batch size large enough such that batch * channels > 65535.
    # For example, use batch=70000 and channels=1.
    batch_size = 70000
    channels = 1
    in_h, in_w = 16, 16
    kernel_h = 3

    x = torch.randn(batch_size, channels, in_h, in_w, device="cuda", dtype=torch.float32)
    # Weight shape: (channels, 1, kernel_h, 1) as expected.
    weight = torch.randn(channels, 1, kernel_h, 1, device="cuda", dtype=torch.float32)
    # Use a bias tensor.
    bias = torch.randn(channels, device="cuda", dtype=torch.float32)

    # Use stride, padding, dilation typical values.
    stride = 1
    padding = 0
    dilation = 1
    groups = channels

    with pytest.raises(RuntimeError):
        # Expect the kernel launch to fail due to grid z dimension overflow.
        out = cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)
        # Synchronize to catch asynchronous errors.
        torch.cuda.synchronize()

# Test Case 2: Kernel Width Generality Issue
# This test feeds a weight tensor with an unexpected kernel width > 1.
# The CUDA kernel is hard-coded for kernel width = 1, so the output
# will be computed incorrectly compared to PyTorch's native depthwise convolution.
def test_kernel_wrong_kernel_width():
    cuda_module = build_kernel()
    batch_size = 8
    channels = 3
    in_h, in_w = 32, 32
    kernel_h = 3
    # Intentionally set kernel_w to 2, which is not supported by the CUDA kernel.
    kernel_w = 2

    x = torch.randn(batch_size, channels, in_h, in_w, device="cuda", dtype=torch.float32)
    # Weight shape: (channels, 1, kernel_h, kernel_w) has an unexpected kernel width.
    weight = torch.randn(channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(channels, device="cuda", dtype=torch.float32)

    stride = 1
    padding = 0
    dilation = 1
    groups = channels

    # Compute a reference output using PyTorch's nn.Conv2d.
    # Note: We'll create a Conv2d layer with groups == channels. The weight must be adjusted.
    conv_ref = nn.Conv2d(
        channels, channels, kernel_size=(kernel_h, kernel_w),
        stride=stride, padding=padding, dilation=dilation, groups=channels, bias=True
    ).cuda()

    # Force the conv_ref to use the same parameters as our weight and bias for consistency.
    with torch.no_grad():
        # The CUDA kernel expects weight shape [channels, 1, kernel_h, _] and bias per channel.
        conv_ref.weight.copy_(weight)
        conv_ref.bias.copy_(bias)

    out_ref = conv_ref(x)

    # Use the CUDA kernel forward. (It will ignore the kernel width > 1.)
    # In our kernel, out_w is computed as if kernel_w == 1.
    out_test = cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)
    torch.cuda.synchronize()

    # The outputs should differ because our kernel is not general to kernel width != 1.
    with pytest.raises(AssertionError):
        # Use an assert that fails if outputs are too close.
        assert torch.allclose(out_ref, out_test, atol=1e-5), (
            "CUDA kernel output unexpectedly matches the reference output "
            "despite unsupported kernel width > 1."
        )
