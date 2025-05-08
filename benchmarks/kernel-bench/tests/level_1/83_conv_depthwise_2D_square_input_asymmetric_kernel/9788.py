
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test case 1: Data type issue – using float64 input should trigger an error.
def test_dtype_issue():
    cuda_module = build_kernel()
    batch, channels, height, width = 2, 3, 8, 8
    kernel_h = 3
    # Create input, weight and bias in double precision (float64)
    x = torch.randn(batch, channels, height, width, dtype=torch.float64, device="cuda")
    # Weight shape: (channels, 1, kernel_h, 1)
    weight = torch.randn(channels, 1, kernel_h, 1, dtype=torch.float64, device="cuda")
    bias = torch.randn(channels, dtype=torch.float64, device="cuda")

    with pytest.raises(RuntimeError):
        # The kernel expects float pointers. This call should raise an error.
        cuda_module.forward(x, weight, bias, stride=1, padding=0, dilation=1, groups=channels)

# Test case 2: Kernel width assumption – using a kernel with width != 1 should yield incorrect results.
def test_non_unit_kernel_width_issue():
    cuda_module = build_kernel()
    batch, channels, height, width = 1, 2, 10, 10
    kernel_h = 3
    kernel_w = 2  # non-unit kernel width, not supported by our kernel

    # Create input of type float32.
    x = torch.randn(batch, channels, height, width, dtype=torch.float32, device="cuda")
    # Create a weight tensor with a second dimension != 1.
    weight = torch.randn(channels, 1, kernel_h, kernel_w, dtype=torch.float32, device="cuda")
    bias = torch.randn(channels, dtype=torch.float32, device="cuda")

    # Run the custom CUDA kernel.
    out_custom = cuda_module.forward(x, weight, bias, stride=1, padding=0, dilation=1, groups=channels)

    # Create a reference convolution using PyTorch's functional conv2d.
    # Note: We set groups=channels to get a depthwise convolution.
    import torch.nn.functional as F
    out_ref = F.conv2d(x, weight, bias=bias, stride=1, padding=0, dilation=1, groups=channels)

    # Since the custom kernel ignores the kernel width (assumes 1 instead), the output will differ.
    # We assert that the maximum difference is significant.
    diff = (out_custom - out_ref).abs().max().item()
    assert diff > 1e-3, f"Expected significant difference due to kernel width assumption, got diff = {diff}"

# Test case 3: Groups parameter check – providing groups != channels should raise an error.
def test_invalid_groups_issue():
    cuda_module = build_kernel()
    batch, channels, height, width = 2, 4, 16, 16
    kernel_h = 3

    x = torch.randn(batch, channels, height, width, dtype=torch.float32, device="cuda")
    weight = torch.randn(channels, 1, kernel_h, 1, dtype=torch.float32, device="cuda")
    bias = torch.randn(channels, dtype=torch.float32, device="cuda")

    # Use an invalid groups parameter (not equal to channels)
    invalid_groups = channels - 1

    with pytest.raises(RuntimeError) as excinfo:
        cuda_module.forward(x, weight, bias, stride=1, padding=0, dilation=1, groups=invalid_groups)
    assert "Depthwise convolution requires groups == number of input channels." in str(excinfo.value)
