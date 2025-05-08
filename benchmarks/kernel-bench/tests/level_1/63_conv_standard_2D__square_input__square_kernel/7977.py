
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="custom_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Helper: run our custom forward function provided by the kernel extension.
def custom_conv_forward(x, weight, bias, stride, padding, dilation, groups):
    module = build_kernel()
    # The kernel provided is exposed as "forward" in the module.
    return module.forward(x, weight, bias, stride, padding, dilation, groups)

# Issue 1: Non–float32 input types are not supported.
def test_non_float_input():
    # Create a double-precision input and weight.
    batch_size = 1
    in_channels = 3
    out_channels = 8
    height = width = 32
    kernel_size = 3
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.double)
    # Note that for a standard conv2d, weight shape is (out_channels, in_channels, kernel_size, kernel_size)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.double)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.double)
    stride, padding, dilation, groups = 1, 0, 1, 1

    with pytest.raises(RuntimeError):
        # Expect the kernel check to fail since it calls CHECK_CUDA and CHECK_CONTIGUOUS only,
        # but UINT_PTR conversion from double to float pointer is not allowed.
        custom_conv_forward(x, weight, bias, stride, padding, dilation, groups)

# Issue 2: Group convolution is not supported.
def test_group_convolution():
    batch_size = 1
    in_channels = 4
    out_channels = 4
    height = width = 16
    kernel_size = 3
    groups = 2  # Group convolution: each group has 2 in-channels.
    stride, padding, dilation = 1, 1, 1

    # For a grouped convolution the weight shape should be (out_channels, in_channels/groups, kH, kW).
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)

    # Expected result using PyTorch built-in conv2d (which supports groups) 
    ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    # The custom kernel ignores groups, so we need to simulate its use by “lifting” weight to full in_channels.
    # Here we call our custom kernel with groups value, but it will perform an ungrouped convolution.
    out = custom_conv_forward(x, weight, bias, stride, padding, dilation, groups)
    # The outputs should differ because our kernel treated the weight as having shape
    # (out_channels, in_channels, kH, kW) while the actual weight has shape corresponding to grouped conv.
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(out, ref, atol=1e-5)
        
# Issue 3: Only square kernels are supported.
def test_non_square_kernel():
    batch_size = 1
    in_channels = 3
    out_channels = 8
    height = width = 32
    # Create a non-square kernel; for example, kernel height != kernel width.
    kernel_h, kernel_w = 3, 5
    stride, padding, dilation, groups = 1, 0, 1, 1

    # PyTorch’s conv2d expects weight shape (out_channels, in_channels, kernel_h, kernel_w)
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    
    # The custom kernel determines kernel_size from weight.size(2) (i.e. kernel_h) and then uses that for both dims.
    # Thus, a non-square kernel will be misinterpreted.
    ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    out = custom_conv_forward(x, weight, bias, stride, padding, dilation, groups)
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(out, ref, atol=1e-5)

# Issue 4: Lack of native batch support (kernel launched per sample on CPU loop)
def test_multiple_batches():
    # Although the kernel iterates over the batch dimension on the host,
    # this design may be inefficient. However, it should still produce correct results.
    batch_size = 8
    in_channels = 3
    out_channels = 16
    height = width = 64
    kernel_size = 3
    stride, padding, dilation, groups = 1, 1, 1, 1

    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)

    ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    out = custom_conv_forward(x, weight, bias, stride, padding, dilation, groups)
    
    # Even though the batch is processed in a loop on the CPU,
    # the numerical result is still expected to be correct.
    torch.testing.assert_allclose(out, ref, atol=1e-5)
