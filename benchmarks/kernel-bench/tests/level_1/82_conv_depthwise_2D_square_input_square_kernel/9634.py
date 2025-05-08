
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_grid():
    # Issue 1: The grid's z–dimension is computed as batch_size * in_channels.
    # This test forces the z–grid to exceed the typical CUDA limit (e.g. 65,535) and should trigger a launch error.
    batch_size = 300
    in_channels = 256  # 300*256 = 76800 > 65535 (typical limit)
    height, width = 64, 64
    kernel_size = 3
    stride = 1
    padding = 1

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # For depthwise conv weight shape, our kernel expects a flattened weight of shape (in_channels, kernel_size, kernel_size)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    module = build_kernel()
    with pytest.raises(RuntimeError):
        # The large grid should trigger a runtime error if z dimension exceeds hardware limits.
        module.forward(x, weight.view(in_channels, kernel_size, kernel_size), bias, stride, padding, groups=in_channels)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ignored_groups():
    # Issue 2: The kernel ignores the groups parameter.
    # Here we create a grouped convolution with groups != in_channels,
    # so the expected output from PyTorch's conv2d will differ from the kernel's depthwise (per–channel) computation.
    batch_size = 4
    in_channels = 4
    height, width = 16, 16
    kernel_size = 3
    stride = 1
    padding = 1
    groups = 2  # Not equal to in_channels, so it's not depthwise.

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # For a grouped convolution, weight shape is (out_channels, in_channels/groups, kernel_size, kernel_size)
    out_channels = 4
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    module = build_kernel()
    # Our kernel, however, expects a weight layout of (in_channels, kernel_size, kernel_size) for depthwise conv.
    # We force a reinterpretation of the weight tensor to mimic that layout.
    weight_for_kernel = weight.view(in_channels, kernel_size, kernel_size)

    out_kernel = module.forward(x, weight_for_kernel, bias, stride, padding, groups)
    out_ref = torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=padding, groups=groups)
    # Because the kernel ignores groups, its output will not match the reference.
    with pytest.raises(AssertionError):
        assert torch.allclose(out_kernel, out_ref, atol=1e-5), \
            "Kernel output unexpectedly matches reference output for grouped convolution!"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_multiple_output_dimensions():
    # Issues 3, 4, and 5: Test the kernel with input dimensions
    # that do not align with the fixed block size and assumed weight layout.
    # This test verifies that the kernel returns an incorrect (or at least inconsistent) result
    # compared to PyTorch's native implementation.
    batch_size = 2
    in_channels = 3
    height, width = 45, 37  # Non–multiple of 32 or 16.
    kernel_size = 3
    stride = 1
    padding = 1

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    module = build_kernel()
    # Pass weight with view to match the kernel's expected (in_channels, kernel_size, kernel_size) layout.
    out_kernel = module.forward(x, weight.view(in_channels, kernel_size, kernel_size), bias, stride, padding, groups=in_channels)
    
    # Expected output using PyTorch's native depthwise conv2d
    conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=True)
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)
    out_ref = conv(x)
    # If the kernel were fully general, the outputs would match.
    # Here we expect a discrepancy due to the fixed block configuration and weight layout assumptions.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), \
        "Kernel output unexpectedly matches reference output for non–multiple output dimensions!"
