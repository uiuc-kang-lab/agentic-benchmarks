
import torch
import pytest
from torch.utils.cpp_extension import load

# Build the kernel extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="conv_transpose2d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Utility to compare against PyTorch native conv_transpose2d.
def pytorch_conv_transpose2d(input, weight, bias, stride, padding, dilation, groups=1):
    return torch.nn.functional.conv_transpose2d(
        input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups
    )

# Issue 1 test: Non-square (rectangular) kernel not supported.
def test_rectangular_kernel():
    # Setup: use rectangular kernel dimensions (height != width)
    batch_size = 2
    in_channels = 4
    out_channels = 3
    in_height = 8
    in_width = 16
    kernel_h = 3
    kernel_w = 5      # rectangular: different from kernel_h
    stride = 2
    padding = 1
    dilation = 1

    # Create input tensor.
    x = torch.randn(batch_size, in_channels, in_height, in_width, device="cuda", dtype=torch.float32)
    # Create weight tensor in conv_transpose2d the expected weight shape is [in_channels, out_channels, kernel_h, kernel_w]
    weight = torch.randn(in_channels, out_channels, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Run the native PyTorch conv_transpose2d
    native_out = pytorch_conv_transpose2d(x, weight, bias, stride, padding, dilation)
    
    # Run our CUDA kernel extension.
    kernel_module = build_kernel()
    # Our kernel wrapper expects a single int for kernel_size, so it will use weight.size(2) as kernel_size.
    # Thus it will assume a square kernel of size kernel_h (3 in this case) and ignore the extra columns.
    ext_out = kernel_module.forward(x, weight, bias, stride, padding, dilation)

    # Because the extension ignores the second dimension of the kernel, the output will differ.
    assert not torch.allclose(ext_out, native_out, atol=1e-4), \
        "Test failed: Extension output unexpectedly matches native output for a rectangular kernel."

# Issue 2 test: Asymmetric stride/padding/dilation not supported.
def test_asymmetric_parameters():
    # Setup: Using different effective parameters for height and width.
    # Note: Our extension accepts single int values whereas PyTorch supports tuple parameters.
    # Therefore, we simulate the case by constructing different output sizes.
    batch_size = 2
    in_channels = 4
    out_channels = 3
    in_height = 10
    in_width = 20
    kernel_size = 3
    # Let’s pick stride, padding, dilation for height and width differently.
    stride_h, stride_w = 2, 3
    padding_h, padding_w = 1, 2
    dilation_h, dilation_w = 1, 2

    x = torch.randn(batch_size, in_channels, in_height, in_width, device="cuda", dtype=torch.float32)
    # Create weight tensor (square kernel as required by the kernel) 
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    # Native conv_transpose2d using asymmetric (tuple) parameters:
    native_out = torch.nn.functional.conv_transpose2d(
        x, weight, bias=bias, stride=(stride_h, stride_w),
        padding=(padding_h, padding_w), dilation=(dilation_h, dilation_w)
    )
    
    # Our extension only accepts one int per parameter.
    # We choose the height values (stride_h, padding_h, dilation_h) for the extension.
    kernel_module = build_kernel()
    ext_out = kernel_module.forward(x, weight, bias, stride_h, padding_h, dilation_h)
    
    # Because the extension ignores the asymmetry in the width direction,
    # the output dimensions and values will differ.
    assert ext_out.shape != native_out.shape or not torch.allclose(ext_out, native_out, atol=1e-4), \
        "Test failed: Extension output unexpectedly matches native output for asymmetric parameters."

# Issue 3 test: Grouped convolutions not supported.
def test_grouped_convolution():
    # Setup: Use a grouped convolution (groups > 1)
    batch_size = 2
    in_channels = 4
    groups = 2
    # For grouped convolution, out_channels must be a multiple of groups.
    out_channels = 4
    in_height = 8
    in_width = 8
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1

    # Create input tensor.
    x = torch.randn(batch_size, in_channels, in_height, in_width, device="cuda", dtype=torch.float32)
    # For grouped conv, weight shape is normally [in_channels, out_channels // groups, kH, kW]
    # However, our extension expects weight of shape [in_channels, out_channels, k, k]
    # We create a weight tensor that “mimics” a grouped convolution but without any grouping logic.
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Native grouped conv_transpose2d:
    native_out = torch.nn.functional.conv_transpose2d(
        x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups
    )

    kernel_module = build_kernel()
    ext_out = kernel_module.forward(x, weight, bias, stride, padding, dilation)
    
    # Because the kernel does not implement groups, the two outputs will differ.
    assert ext_out.shape != native_out.shape or not torch.allclose(ext_out, native_out, atol=1e-4), \
        "Test failed: Extension output unexpectedly matches native output for grouped convolution."

if __name__ == "__main__":
    pytest.main([__file__])
