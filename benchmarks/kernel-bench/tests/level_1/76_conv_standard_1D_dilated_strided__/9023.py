
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build the CUDA extension module from kernel.cu
def build_kernel():
    module = load(
        name="test_conv1d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return module

# Issue 1: Kernel only supports float32.
# Test: Pass an input tensor with a different data type (e.g. float64)
def test_kernel_input_dtype():
    my_module = build_kernel()
    # Create double precision (float64) inputs
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = 3
    length = 20
    stride = 1
    dilation = 1

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float64)
    # Weight: shape (out_channels, in_channels, kernel_size)
    weight = torch.randn(out_channels, in_channels, kernel_size, device="cuda", dtype=torch.float64)
    # bias is optional, here omit it.
    # Expect that using a non-float32 tensor will lead to an error since kernel casts
    with pytest.raises(RuntimeError):
        # The forward function inside the module expects x and weight to be float32.
        _ = my_module.forward(x, weight, None, stride, dilation)


# Issue 2: Kernel does not support convolution padding.
# Test: Compare the custom kernel output with a reference convolution that uses padding.
def test_kernel_padding_mismatch():
    my_module = build_kernel()
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = 3
    length = 32
    stride = 2
    dilation = 1
    padding = 1  # typical padding value

    # Create input tensor and weight in proper float32 type
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = None

    # Compute output using the custom kernel
    custom_out = my_module.forward(x, weight, bias, stride, dilation)
    # Compute reference convolution using torch.nn.functional.conv1d with padding.
    # Note: The custom kernel does NOT take padding into account.
    ref_out = torch.nn.functional.conv1d(x, weight, bias=bias, stride=stride, dilation=dilation, padding=padding)

    # The outputs should differ because the custom kernel computes valid convolution (padding = 0)
    # while the reference convolution uses padding=1.
    # We use .allclose with a very loose tolerance to ensure they are not (accidentally) equal.
    is_close = torch.allclose(custom_out, ref_out, atol=1e-6)
    assert not is_close, (
        "Kernel output unexpectedly matches the padded convolution output. "
        "This indicates that the kernel is erroneously handling padding."
    )
