
import pytest
import torch
from torch import nn
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Test with a non-float32 tensor (e.g. float64)
# Expect that the kernel produces numerically incorrect results when given an unsupported data type.
def test_incorrect_dtype():
    my_module = build_kernel()
    # Create input, weight and bias as float64 (double) tensors.
    batch_size = 4
    in_channels = 3
    out_channels = 8
    height = 32
    width = 32
    kernel_h, kernel_w = 3, 3

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float64)
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, device="cuda", dtype=torch.float64)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float64)

    # Run the custom kernel (which expects float32).
    out_kernel = my_module.forward(x, weight, bias, 1, 0, 1, 1)
    
    # Build a reference convolution (casting inputs to float32 so that nn.Conv2d works naturally)
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_h, kernel_w), stride=1, padding=0, bias=True).cuda()
    # Copy weights and bias (note the dtype difference)
    conv.weight.data.copy_(weight.float())
    conv.bias.data.copy_(bias.float())

    # Run the reference convolution with casted inputs.
    out_ref = conv(x.float())
    
    # Since the kernel used input memory as float32 even though provided double,
    # we expect a significant mismatch.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-3), \
        "Kernel accepted non-float32 tensor without error, but should not produce correct results."

# Issue 2: Test with non-uniform (tuple) stride/padding/dilation.
# The kernel only accepts an integer for these parameters. Passing a tuple should lead to an error.
def test_non_uniform_conv_params():
    my_module = build_kernel()
    batch_size = 4
    in_channels = 3
    out_channels = 8
    height = 32
    width = 32
    kernel_h, kernel_w = 3, 3

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Try passing a tuple for stride. The extension forward is defined to receive an int.
    with pytest.raises(TypeError):
        # This should fail because a tuple cannot be converted to int implicitly.
        my_module.forward(x, weight, bias, (1, 2), 0, 1, 1)

    # Similarly, passing tuple for padding/dilation should also raise an error.
    with pytest.raises(TypeError):
        my_module.forward(x, weight, bias, 1, (0, 1), 1, 1)

    with pytest.raises(TypeError):
        my_module.forward(x, weight, bias, 1, 0, (1, 2), 1)

# Issue 3: Test for grid dimension limits.
# Create a situation where batch_size*out_channels exceeds the maximum allowed grid dimension in z.
def test_grid_dimension_limit():
    my_module = build_kernel()
    # Set up parameters so that batch_size * out_channels > 65,535.
    batch_size = 1
    # Choose out_channels large enough to exceed the z-dimension limit. (e.g. 70,000)
    out_channels = 70000
    in_channels = 1
    height = 8
    width = 8
    kernel_h, kernel_w = 1, 1

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # With such a large grid size in the z-dimension the kernel launch should fail.
    with pytest.raises(RuntimeError):
        out_kernel = my_module.forward(x, weight, bias, 1, 0, 1, 1)
        # Synchronize to force any launch errors.
        torch.cuda.synchronize()

# Issue 4: Test when the number of input channels per group is not a multiple of 4.
# The fixed loop unrolling may not be optimal or could introduce issues.
def test_loop_unroll_with_non_multiple_of_four():
    my_module = build_kernel()
    batch_size = 2
    in_channels = 3   # 3 is not a multiple of 4.
    out_channels = 6
    height = 16
    width = 16
    kernel_h, kernel_w = 3, 3

    # Use groups=1 so that in_channels per group = 3.
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Compute output using the custom kernel.
    out_kernel = my_module.forward(x, weight, bias, 1, 1, 1, 1)

    # Compute output using PyTorch nn.Conv2d.
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_h, kernel_w), stride=1, padding=1, bias=True).cuda()
    conv.weight.data.copy_(weight)
    conv.bias.data.copy_(bias)
    out_ref = conv(x)

    # We expect the outputs to disagree significantly because of potential issues induced
    # by the fixed unrolling factor when channels are not a multiple of 4.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-3), \
        "Kernel output unexpectedly matches reference even with non-multiple of four input channels."
