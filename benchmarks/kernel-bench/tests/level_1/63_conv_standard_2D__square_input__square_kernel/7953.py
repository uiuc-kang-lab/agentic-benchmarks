
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Helper to compile and load the CUDA extension from kernel.cu
def build_kernel():
    module = load(
        name="custom_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: Kernel assumes float32
def test_non_float32_dtype():
    # Build module
    conv_module = build_kernel()

    # Create tensors with float64 instead of float32
    batch_size = 2
    in_channels = 3
    out_channels = 8
    height = width = 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    bias_flag = False

    # Use torch.float64 input, weight (and bias if needed)
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float64, device="cuda")
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float64, device="cuda")
    # Our kernel will interpret data as float32 so results will be wrong.
    out_custom = conv_module.forward(x, weight, None, stride, padding, dilation, 1)
    # Get reference output using F.conv2d (which will do proper conversion if needed)
    x32 = x.float()
    weight32 = weight.float()
    out_ref = F.conv2d(x32, weight32, bias=None, stride=stride, padding=padding, dilation=dilation)
    # The outputs should differ because the kernel misinterprets float64 data.
    assert not torch.allclose(out_custom, out_ref, atol=1e-5), (
        "Kernel accepted float64 input without error, but outputs match F.conv2d unexpectedly."
    )

# Issue 2: Only groups == 1 are supported
def test_groups_not_one():
    conv_module = build_kernel()

    batch_size = 2
    in_channels = 4
    out_channels = 8
    height = width = 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    
    # Attempt to use groups != 1 should trigger a runtime check
    with pytest.raises(RuntimeError, match="Only groups=1 is supported"):
        conv_module.forward(x, weight, None, stride, padding, dilation, groups=2)

# Issue 3: Weight tensor size exceeds constant memory capacity
def test_weight_constant_memory_overflow():
    conv_module = build_kernel()

    batch_size = 2
    in_channels = 16
    out_channels = 16
    # Choose kernel dimensions so that weight.numel() > 16384.
    # For example, kernel size 9: 9*9*16*16 = 20736 > 16384.
    kernel_size = 9
    stride = 1
    padding = 4
    dilation = 1

    x = torch.randn(batch_size, in_channels, 32, 32, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="Weight tensor too large for constant memory"):
        conv_module.forward(x, weight, None, stride, padding, dilation, groups=1)

# Issue 4: Kernel assumes square convolution kernels
def test_non_square_kernel():
    conv_module = build_kernel()

    batch_size = 2
    in_channels = 3
    out_channels = 8
    height = width = 16
    # Create a non-square kernel: height=3, width=5.
    # The kernel code uses weight.size(2) as kernel_size, so this mismatch will lead to incorrect indexing.
    kernel_height = 3
    kernel_width = 5
    stride = 1
    padding = 1
    dilation = 1

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Manually create a non-square weight.
    weight = torch.randn(out_channels, in_channels, kernel_height, kernel_width, device="cuda", dtype=torch.float32)

    # Get output from our custom kernel.
    out_custom = conv_module.forward(x, weight, None, stride, padding, dilation, groups=1)
    # Get reference output via F.conv2d (which supports non-square kernels).
    out_ref = F.conv2d(x, weight, bias=None, stride=stride, padding=padding, dilation=dilation)
    
    # The outputs will differ due to the kernel's square-kernel assumption.
    assert not torch.allclose(out_custom, out_ref, atol=1e-4), (
        "Kernel incorrectly handled non-square kernel parameters; outputs match F.conv2d unexpectedly."
    )

# Issue 5: Lack of kernel launch error checking may hide launch configuration problems.
def test_kernel_launch_without_error_checking():
    conv_module = build_kernel()

    # Create parameters that lead to an invalid output shape.
    # For example, choose dilation such that computed output height/width becomes zero or negative.
    batch_size = 1
    in_channels = 3
    out_channels = 8
    height = width = 10
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 20  # Excessively high dilation causing negative effective receptive field

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)

    # F.conv2d would raise an error or return an empty tensor.
    # Our custom kernel does not check for such misconfigurations, likely launching a kernel with a grid that does not cover any valid site.
    out_custom = conv_module.forward(x, weight, None, stride, padding, dilation, groups=1)
    # We can check if the output has a shape that does not make sense.
    out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    expected_shape = (batch_size, out_channels, out_height, out_width)
    assert out_custom.shape == expected_shape, (
        f"Kernel output shape {out_custom.shape} does not match expected {expected_shape}. "
        "Lack of proper launch error checking may hide configuration issues."
    )
