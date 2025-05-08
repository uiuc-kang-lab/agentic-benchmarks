
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Utility to build the CUDA extension from kernel.cu.
def build_kernel():
    return load(
        name="depthwise_conv_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: Kernel width hardcoded to 1.
# This test uses a weight with kernel_width > 1 and expects the CUDA extension result
# to differ from the reference convolution computed by torch.nn.functional.conv2d.
def test_kernel_width_support():
    device = "cuda"
    batch_size, channels = 2, 3
    in_h, in_w = 16, 16
    # Use a 3x3 kernel, which the extension cannot support properly.
    kernel_h, kernel_w = 3, 3
    stride = 1
    padding = 1
    dilation = 1

    # Create input tensor.
    x = torch.randn(batch_size, channels, in_h, in_w, device=device, dtype=torch.float32)
    # Manually create a weight tensor with a kernel width > 1.
    # For depthwise convolution, weight shape: (channels, 1, kernel_h, kernel_w)
    weight = torch.randn(channels, 1, kernel_h, kernel_w, device=device, dtype=torch.float32)
    # Create bias tensor.
    bias = torch.randn(channels, device=device, dtype=torch.float32)

    # Compute reference using conv2d.
    ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=(dilation, dilation), groups=channels)

    # Call the CUDA kernel (which expects kernel_w == 1)
    # We simulate passing a weight that has the wrong shape (flattened as if kernel_w==1).
    # Here we artificially take only the central column of the kernel.
    weight_incorrect = weight[:,:, :, kernel_w // 2].contiguous()  # shape: (channels, 1, kernel_h)
    # Adjust weight_incorrect view to (channels, kernel_h)
    weight_incorrect = weight_incorrect.view(channels, kernel_h)

    # The extension expects weight shape as (channels * kernel_h) flattened.
    cuda_kernel = build_kernel()
    out = cuda_kernel.forward(x, weight_incorrect, bias, stride, padding, dilation, channels)

    # The output from the CUDA kernel should differ from the correct result.
    assert not torch.allclose(out, ref, atol=1e-5), "Test failed: Kernel output unexpectedly matches reference despite wrong kernel width."

# Issue 2: Only supports float32.
# This test calls the kernel with double precision inputs expecting it to fail.
def test_dtype_support():
    device = "cuda"
    batch_size, channels = 2, 3
    in_h, in_w = 16, 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1

    x = torch.randn(batch_size, channels, in_h, in_w, device=device, dtype=torch.float64)
    weight = torch.randn(channels, 1, kernel_size, device=device, dtype=torch.float64)
    # Flatten weight as expected by kernel.
    weight = weight.view(channels, kernel_size)
    bias = torch.randn(channels, device=device, dtype=torch.float64)

    cuda_kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect a runtime error because the kernel will misinterpret the data (due to type mismatch).
        cuda_kernel.forward(x, weight, bias, stride, padding, dilation, channels)

# Issue 3: Output width computation does not account for dilation in the width dimension.
# This test uses a dilation > 1 and observes a mismatch in output shape or result.
def test_dilation_width_issue():
    device = "cuda"
    batch_size, channels = 2, 3
    in_h, in_w = 32, 32
    kernel_size = 3  # kernel height; extension always assumes kernel width==1.
    stride = 1
    padding = 1
    dilation = 2  # dilation greater than 1

    x = torch.randn(batch_size, channels, in_h, in_w, device=device, dtype=torch.float32)
    # Create a weight tensor for a kernel with a single width dimension.
    weight = torch.randn(channels, 1, kernel_size, device=device, dtype=torch.float32)
    weight_incorrect = weight.view(channels, kernel_size)
    bias = torch.randn(channels, device=device, dtype=torch.float32)

    # Compute reference using conv2d.
    # For width, proper dilation handling would require: effective kernel width = dilation*(1-1)+1 = 1,
    # but the general formula for width should use dilation if kernel width > 1.
    # To simulate the general case, we consider a convolution with kernel width 1 but variation in dilation:
    ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=(dilation, dilation), groups=channels)

    cuda_kernel = build_kernel()
    out = cuda_kernel.forward(x, weight_incorrect, bias, stride, padding, dilation, channels)

    # We expect a discrepancy because the kernel's output width is computed ignoring dilation for width.
    assert not torch.allclose(out, ref, atol=1e-5), "Test failed: Kernel output unexpectedly matches reference when dilation width issue should occur."

# Issue 4: The kernel only supports a uniform stride and dilation for both height and width.
# This test attempts to mimic a scenario with different effective stride requirements.
# Since the extension takes a single stride value, we force a situation where a non‐uniform treatment would be expected.
def test_non_uniform_stride_issue():
    device = "cuda"
    batch_size, channels = 2, 3
    in_h, in_w = 32, 64  # non-square input to emphasize stride differences
    kernel_size = 3  # kernel height; kernel width is assumed to be 1 in the kernel.
    stride = 2  # uniform stride given, but the impact on width is problematic if a general conv were expected.
    padding = 1
    dilation = 1

    x = torch.randn(batch_size, channels, in_h, in_w, device=device, dtype=torch.float32)
    weight = torch.randn(channels, 1, kernel_size, device=device, dtype=torch.float32)
    weight_incorrect = weight.view(channels, kernel_size)
    bias = torch.randn(channels, device=device, dtype=torch.float32)

    # Compute reference using conv2d configured for a 1-width kernel.
    ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=(dilation, dilation), groups=channels)

    cuda_kernel = build_kernel()
    out = cuda_kernel.forward(x, weight_incorrect, bias, stride, padding, dilation, channels)

    # Because the kernel ignores potential non-uniformities needed for generalized stride,
    # the results are expected to be different.
    assert not torch.allclose(out, ref, atol=1e-5), "Test failed: Kernel output unexpectedly matches reference for non-uniform stride scenario."
