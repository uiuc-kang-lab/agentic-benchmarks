
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper to build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel width hard-coded to 1.
# This test creates a convolution weight with a kernel width != 1.
# We use a normal PyTorch convolution as reference. Their outputs should differ.
def test_kernel_width_incompatibility():
    cuda_module = build_kernel()

    batch = 2
    channels = 3
    in_h = 16
    in_w = 16
    kernel_h = 3
    kernel_w = 3  # NOT 1, which the kernel does not support

    # Create an input and a weight tensor. Note that our CUDA kernel expects weight shape
    # as if it were [channels, 1, kernel_h, 1]. Here we deliberately create a weight with width 3.
    x = torch.randn(batch, channels, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.randn(channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(channels, device="cuda", dtype=torch.float32)

    # Compute our custom kernel output
    # Our forward function assumes:
    #   weight size: [channels, 1, kernel_h, 1]
    #   kernel width of 1.
    # We pass the unmodified weight, so our kernel will ignore the extra width.
    out_custom = cuda_module.forward(x, weight, bias, stride=1, padding=0, dilation=1, groups=channels)

    # Create a reference convolution with nn.Conv2d that uses the full weight dimensions.
    conv_ref = torch.nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=(kernel_h, kernel_w),
        stride=1,
        padding=0,
        dilation=1,
        groups=channels,
        bias=True,
    ).cuda().to(dtype=torch.float32)

    # Overwrite the weights and bias of conv_ref with our tensors.
    # Note: conv_ref.weight is of shape (channels, 1, kernel_h, kernel_w)
    with torch.no_grad():
        conv_ref.weight.copy_(weight)
        conv_ref.bias.copy_(bias)

    out_ref = conv_ref(x)

    # Since our CUDA kernel ignores the kernel width > 1, its output should differ.
    assert not torch.allclose(out_custom, out_ref, atol=1e-4), (
        "Custom kernel output unexpectedly matches reference output even though kernel width > 1 "
        "is not handled correctly."
    )

# Issue 2: Only supports float32 inputs.
# This test creates doubles as input to trigger the type incompatibility.
def test_non_float32_input():
    cuda_module = build_kernel()

    batch = 2
    channels = 3
    in_h = 16
    in_w = 16
    kernel_h = 3

    # Create input, weight, and bias of type float64.
    x = torch.randn(batch, channels, in_h, in_w, device="cuda", dtype=torch.float64)
    weight = torch.randn(channels, 1, kernel_h, 1, device="cuda", dtype=torch.float64)
    bias = torch.randn(channels, device="cuda", dtype=torch.float64)

    # The CUDA kernel only supports float32, so we expect a runtime error due to type mismatch.
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(x, weight, bias, stride=1, padding=0, dilation=1, groups=channels)

# Issue 3: The kernel uses a single scalar for stride, padding, and dilation,
# which limits asymmetric convolution operations.
# This test passes parameters that would be interpreted as asymmetric in a general conv2d,
# and compares against a reference. The outputs should differ.
def test_asymmetric_convparams():
    cuda_module = build_kernel()

    batch = 2
    channels = 3
    in_h = 20
    in_w = 20
    kernel_h = 3

    # Create an input tensor.
    x = torch.randn(batch, channels, in_h, in_w, device="cuda", dtype=torch.float32)

    # For the custom kernel, we assume the kernel width is 1. However, we simulate an asymmetric scenario
    # by selecting stride and padding that would be different for height and width in a general case.
    # In this test we treat the scalar arguments as if they applied to both dimensions.
    stride = 2   # Intended for height only.
    padding = 1  # Intended for height only.
    dilation = 1

    # Set up weight and bias.
    weight = torch.randn(channels, 1, kernel_h, 1, device="cuda", dtype=torch.float32)
    bias = torch.randn(channels, device="cuda", dtype=torch.float32)

    # Get output from our custom kernel.
    out_custom = cuda_module.forward(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=channels)

    # Create a reference convolution where height and width parameters can differ.
    # Here, we mimic an asymmetric operation by using different kernel sizes for height and width.
    conv_ref = torch.nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=(kernel_h, 1),  # only height is kernel_h, width is 1
        stride=(stride, 1),         # height gets stride, width uses default stride 1
        padding=(padding, 0),       # height gets padding, width gets no padding
        dilation=(dilation, 1),
        groups=channels,
        bias=True,
    ).cuda().to(dtype=torch.float32)

    # Overwrite weights to match our custom kernel weight.
    with torch.no_grad():
        conv_ref.weight.copy_(weight)
        conv_ref.bias.copy_(bias)

    out_ref = conv_ref(x)

    # Because our CUDA kernel uses the same scalar for both dimensions, its output will differ
    # from the correctly computed asymmetric reference.
    assert not torch.allclose(out_custom, out_ref, atol=1e-4), (
        "Custom kernel output unexpectedly matches the asymmetric reference output. "
        "This indicates that the kernel is not handling asymmetric convolution parameters correctly."
    )

if __name__ == "__main__":
    pytest.main([__file__])
