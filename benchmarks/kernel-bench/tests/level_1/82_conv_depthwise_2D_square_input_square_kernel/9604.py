
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu
    module = load(
        name="custom_depthwise",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: Double precision not handled correctly for shared memory allocation.
def test_double_dtype():
    # Use double precision input, weight and bias.
    batch_size = 2
    in_channels = 3
    height = width = 32
    kernel_size = 3
    stride = 1
    padding = 1

    # Create random double precision tensors on CUDA.
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.double)
    # For depthwise conv, weight shape = (in_channels, 1, kernel_size, kernel_size)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.double)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.double)

    # Expected output computed via PyTorch native functional depthwise conv2d.
    ref = torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=padding, groups=in_channels)

    # Call custom kernel.
    custom_module = build_kernel()
    # The extension expects weight in shape (in_channels, kernel_size, kernel_size)
    weight_reshaped = weight.view(in_channels, kernel_size, kernel_size)
    y = custom_module.forward(x, weight_reshaped, bias, stride, padding, in_channels)

    # Because of the shared memory allocation issue, we expect a difference.
    assert not torch.allclose(y, ref, atol=1e-5), "Double precision input should trigger a shared memory allocation issue."

# Issue 2: Kernel ignores the groups argument so it fails for non-depthwise convolutions.
def test_non_depthwise_groups():
    # Construct a convolution where groups != in_channels.
    batch_size = 2
    in_channels = 4  # number of input channels
    groups = 2       # groups set to 2 instead of in_channels (which would be 4 for depthwise)
    height = width = 32
    kernel_size = 3
    stride = 1
    padding = 1

    # Create input tensor.
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)

    # Create weight tensor.
    # For a general group convolution, the weight shape is (out_channels, in_channels/groups, kernel_size, kernel_size)
    # Here, let out_channels == in_channels for simplicity.
    weight = torch.randn(in_channels, in_channels//groups, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    # Expected output computed via PyTorch native conv2d.
    ref = torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=padding, groups=groups)

    custom_module = build_kernel()
    # The custom kernel expects weight in shape (in_channels, kernel_size, kernel_size).
    # We simulate depthwise-like weight arrangement, so we force weight to be depthwise.
    # This test deliberately passes a weight tensor that does not match the group setting.
    weight_wrong = weight.view(in_channels, kernel_size, kernel_size)  # Incorrect reshape for non-depthwise conv.

    y = custom_module.forward(x, weight_wrong, bias, stride, padding, groups)
    # Since the custom kernel ignores groups, its output will be from a depthwise assumption.
    # We check that the output deviates from the reference.
    assert not torch.allclose(y, ref, atol=1e-5), "Non-depthwise convolution (groups != in_channels) should trigger an indexing issue."

# Issue 3: Use of #pragma unroll with a runtime-dependent kernel_size.
def test_runtime_kernel_size():
    # Use a kernel_size that is not typical (and not a compile-time constant) to see potential unroll issues.
    batch_size = 2
    in_channels = 3
    height = width = 64
    kernel_size = 5   # non-standard size that may not be unrolled properly
    stride = 1
    padding = 2

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    ref = torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=padding, groups=in_channels)

    custom_module = build_kernel()
    weight_reshaped = weight.view(in_channels, kernel_size, kernel_size)
    y = custom_module.forward(x, weight_reshaped, bias, stride, padding, in_channels)

    # The unrolling pragma may cause performance or computational issues when kernel_size is not compile-time constant.
    # We expect a discrepancy in the output.
    assert not torch.allclose(y, ref, atol=1e-5), "A runtime kernel_size with unroll pragmas should trigger an issue."

if __name__ == '__main__':
    pytest.main([__file__])
