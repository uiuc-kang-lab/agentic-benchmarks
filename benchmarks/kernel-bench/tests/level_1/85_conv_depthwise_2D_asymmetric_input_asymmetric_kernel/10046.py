
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="depthwise_conv_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test case 1: Trigger issue with general grouped convolution (channels_per_group > 1)
def test_grouped_convolution_incorrect_result():
    # Create a case where in_channels=4 and groups=2 rather than depthwise
    batch_size = 1
    in_channels = 4
    groups = 2  # channels_per_group will be 2 (general grouped convolution)
    # spatial dimensions
    H, W = 8, 8
    kernel_h, kernel_w = 3, 3
    stride, padding, dilation = 1, 1, 1

    # Input tensor and weight tensor (PyTorch weight shape for Conv2d: [out_channels, in_channels/groups, kH, kW])
    x = torch.randn(batch_size, in_channels, H, W, device="cuda", dtype=torch.float32)
    # For grouped convolution, let out_channels be groups * (in_channels // groups) = in_channels.
    # Thus weight shape: [in_channels, 1?, ..., but in general, there are (in_channels/groups) channels per group.
    # To test the issue, we provide a weight tensor with shape [out_channels, in_channels//groups, kH, kW] where in_channels//groups > 1.
    weight = torch.randn(in_channels, in_channels // groups, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = None

    # Use the PyTorch reference convolution as ground truth.
    ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    # Call our CUDA kernel. Note: our kernel expects weight with shape interpreted as if channels_per_group==1.
    mod = build_kernel()
    # Our kernel forward signature:
    # forward(x, weight, optional_bias, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups)
    result = mod.forward(x, weight, bias, stride, stride, padding, padding, dilation, dilation, groups)
    torch.cuda.synchronize()

    # Since the kernel does not sum over the extra channel in each group, the result will be incorrect.
    with pytest.raises(AssertionError):
        # We expect the two outputs not to match.
        assert torch.allclose(result, ref, atol=1e-5), (
            f"Expected discrepancy due to incorrect handling of grouped conv, but got close results."
        )

# Test case 2: Trigger grid dimension issue by forcing batch_size * out_channels to be huge.
def test_grid_dimension_limit():
    # Typical CUDA gridDim.z limit is 65535. We try to exceed that.
    # We keep spatial size minimal, but make batch_size * out_channels > 70000.
    # For a depthwise convolution, out_channels == in_channels.
    batch_size = 500
    in_channels = 150  # 500 * 150 = 75000 > 65535
    groups = in_channels  # valid depthwise
    H, W = 4, 4
    kernel_h, kernel_w = 3, 3
    stride, padding, dilation = 1, 1, 1

    x = torch.randn(batch_size, in_channels, H, W, device="cuda", dtype=torch.float32)
    # For depthwise conv, weight shape would be [in_channels, 1, kernel_h, kernel_w]
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = None

    mod = build_kernel()
    # Depending on the device properties, launching the kernel with gridDim.z > limit should raise a CUDA error.
    with pytest.raises(Exception):
        result = mod.forward(x, weight, bias, stride, stride, padding, padding, dilation, dilation, groups)
        torch.cuda.synchronize()

# Test case 3: Trigger issue by passing an input tensor with non-float32 type.
def test_input_dtype_mismatch():
    batch_size = 1
    in_channels = 3
    groups = in_channels
    H, W = 16, 16
    kernel_h, kernel_w = 3, 3
    stride, padding, dilation = 1, 0, 1

    # Create a double tensor instead of float32.
    x = torch.randn(batch_size, in_channels, H, W, device="cuda", dtype=torch.float64)
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float64)
    bias = None

    mod = build_kernel()
    # The kernel expects float pointers. Passing double tensors will lead to incorrect memory interpretation.
    result = mod.forward(x, weight, bias, stride, stride, padding, padding, dilation, dilation, groups)
    torch.cuda.synchronize()

    # Run a reference depthwise convolution using PyTorch (after converting x & weight to float32)
    x_float = x.float()
    weight_float = weight.float()
    ref = F.conv2d(x_float, weight_float, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    # The result from the kernel (which misinterprets the data) should not be equal to the reference.
    with pytest.raises(AssertionError):
        assert torch.allclose(result, ref, atol=1e-5), (
            f"Kernel output should be incorrect due to dtype mismatch, but got close results."
        )
