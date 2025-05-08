
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

# Utility to compile the CUDA extension from kernel.cu
def build_kernel():
    module = load(
        name="depthwise_conv_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Test case 1: Kernel width not handled (Issue 1 and Issue 2)
# We create a weight tensor with kernel width > 1 (e.g., 3) and
# compare the output of our extension (which ignores the width dimension)
# with PyTorch's Conv2d reference. They should differ.
def test_kernel_with_non_unit_width():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    batch_size = 4
    in_channels = 3
    in_h = 32
    in_w = 32
    kernel_h = 3
    kernel_w = 3  # width greater than 1 (non-unit)
    stride = 1
    padding = 1
    dilation = 1

    # Create input and weight. Note: For depthwise convolution in PyTorch,
    # weight shape is: (in_channels, 1, kernel_h, kernel_w)
    x = torch.randn(batch_size, in_channels, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    # Reference conv2d using groups to get depthwise convolution result.
    conv_ref = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_h, kernel_w),
                         stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=True).cuda()
    conv_ref.weight.data.copy_(weight)
    conv_ref.bias.data.copy_(bias)

    ref_output = conv_ref(x)

    # Call the custom CUDA kernel through our extension.
    module = build_kernel()
    # The extension expects:
    # forward(x, weight, bias, stride, padding, dilation, groups)
    ext_output = module.forward(x, weight, bias, stride, padding, dilation, in_channels)
    torch.cuda.synchronize()

    # Since the kernel ignores kernel width, the outputs should mismatch.
    assert not torch.allclose(ext_output, ref_output, atol=1e-5), \
        "Test failed: Kernel output matches reference even though kernel width is not handled."

# Test case 2: Incorrect iw calculation leading to wrong output (Issue 2)
# Using dilation on height (and implicitly width) may expose the error in computing the input index.
def test_incorrect_iw_calculation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    batch_size = 2
    in_channels = 2
    in_h = 28
    in_w = 28
    kernel_h = 3
    kernel_w = 3  # non-unit width
    stride = 1
    padding = 1
    dilation = 2  # using dilation to compound the error

    x = torch.randn(batch_size, in_channels, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    conv_ref = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_h, kernel_w),
                         stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=True).cuda()
    conv_ref.weight.data.copy_(weight)
    conv_ref.bias.data.copy_(bias)

    ref_output = conv_ref(x)
    module = build_kernel()
    ext_output = module.forward(x, weight, bias, stride, padding, dilation, in_channels)
    torch.cuda.synchronize()

    # They should not be equal because the kernel never adjusts iw for kernel width.
    assert not torch.allclose(ext_output, ref_output, atol=1e-5), \
        "Test failed: Incorrect iw calculation did not affect the output as expected."

# Test case 3: Incorrect output width calculation (Issue 3)
# We choose parameters so that the correct output width computed by PyTorch differs from the one
# that the extension computes.
def test_incorrect_outw_calculation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    batch_size = 2
    in_channels = 1
    in_h = 24
    in_w = 24
    kernel_h = 3
    kernel_w = 3  # non-unit width to exacerbate the miscalculation of output dims
    stride = 2
    padding = 1
    dilation = 2

    x = torch.randn(batch_size, in_channels, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)
    
    conv_ref = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_h, kernel_w),
                         stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=True).cuda()
    conv_ref.weight.data.copy_(weight)
    conv_ref.bias.data.copy_(bias)
    ref_output = conv_ref(x)

    module = build_kernel()
    ext_output = module.forward(x, weight, bias, stride, padding, dilation, in_channels)
    torch.cuda.synchronize()

    # The wrong out_w computation should lead to a different output shape.
    assert ext_output.shape != ref_output.shape, \
        "Test failed: The output shapes are identical despite incorrect out_w calculation."

# Test case 4: Passing non-float32 data (Issues regarding data type support)
# The kernel expects float32 inputs. Supplying tensors with a different dtype (e.g., float64) should trigger errors.
def test_non_float_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    batch_size = 2
    in_channels = 3
    in_h = 16
    in_w = 16
    kernel_h = 3
    kernel_w = 3
    stride = 1
    padding = 1
    dilation = 1

    # Create double precision tensors.
    x = torch.randn(batch_size, in_channels, in_h, in_w, device="cuda", dtype=torch.float64)
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float64)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float64)

    module = build_kernel()

    with pytest.raises(RuntimeError):
        # The extension should fail because it is compiled for float32 only.
        _ = module.forward(x, weight, bias, stride, padding, dilation, in_channels)

# Test case 5: Hardcoded block size and tile assumption may lead to errors for non-multiples of tile size.
def test_tile_boundary_conditions():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Choose output dimensions (h, w) that are not multiples of TILE_SIZE (16).
    batch_size = 1
    in_channels = 1
    # For convolution parameters below, the output size will likely be non-multiple of 16.
    in_h = 37
    in_w = 37
    kernel_h = 3
    kernel_w = 3
    stride = 1
    padding = 0
    dilation = 1

    x = torch.randn(batch_size, in_channels, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    conv_ref = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_h, kernel_w),
                         stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=True).cuda()
    conv_ref.weight.data.copy_(weight)
    conv_ref.bias.data.copy_(bias)
    ref_output = conv_ref(x)

    module = build_kernel()
    ext_output = module.forward(x, weight, bias, stride, padding, dilation, in_channels)
    torch.cuda.synchronize()

    # Even if the kernel handles boundary conditions (by checking oh < out_h, etc.), the hardcoded tile/block
    # assumptions will likely result in an incorrect result.
    assert not torch.allclose(ext_output, ref_output, atol=1e-5), \
        "Test failed: Tile boundary assumptions did not lead to an output difference as expected."
