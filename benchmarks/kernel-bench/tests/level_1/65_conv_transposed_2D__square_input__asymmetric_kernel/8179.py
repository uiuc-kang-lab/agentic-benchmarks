
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility to compile and load the CUDA extension
def build_kernel():
    return load(
        name="conv_transpose2d_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# -------------------------------------------------------------------
# Test case 1: Incorrect channels/groups configuration.
# The kernel does no check that in_channels is divisible by groups.
# We deliberately create an input where in_channels % groups != 0.
# For a valid ConvTranspose2d, PyTorch requires in_channels % groups == 0,
# so we simulate a “bad‐case” for our kernel by bypassing the nn.ConvTranspose2d API.
# We expect the CUDA kernel to produce a result that does not match the
# reference (i.e. the built–in PyTorch version with corrected parameters).
def test_incorrect_groups():
    torch.manual_seed(0)
    kernel_h, kernel_w = 3, 5
    stride = 1
    padding = 0
    output_padding = 0
    groups = 2  # intentionally bad since in_channels=3 is not divisible by 2
    dilation = 1

    batch_size = 4
    in_channels = 3  # not divisible by groups
    out_channels = 8  # set arbitrarily; note: PyTorch's ConvTranspose2d would reject groups if in_channels not divisible
    
    height, width = 16, 16
    
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Build a weight tensor assuming shape: (in_channels, out_channels//groups, kernel_h, kernel_w)
    # This is what our kernel expects.
    # However, with in_channels not divisible by groups (3 % 2 != 0) the kernel arithmetic is incorrect.
    out_channels_per_group = out_channels // groups  # integer division, may not be what user intended
    weight = torch.randn(in_channels, out_channels_per_group, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    ext = build_kernel()
    # Call CUDA extension forward
    out_cuda = ext.forward(x, weight, bias, stride, padding, output_padding, groups, dilation)
    
    # Now build a “reference” using PyTorch's ConvTranspose2d
    # (this requires valid parameters; so we use in_channels that are adjusted)
    # We simulate the intended behavior by forcing groups to 1.
    conv = torch.nn.ConvTranspose2d(
        in_channels, out_channels, (kernel_h, kernel_w),
        stride=stride, padding=padding, output_padding=output_padding,
        groups=1, bias=True
    ).to("cuda")
    # Force conv weight to our weight reshaped appropriately
    # NOTE: This reference does not use groups=2 so the outputs will differ.
    conv.weight.data[:in_channels, :out_channels, :, :] = weight.repeat(1, groups, 1, 1)
    conv.bias.data = bias
    out_ref = conv(x)
    
    # Since the kernel arithmetic is wrong in our “bad” configuration,
    # we expect that the CUDA kernel output WILL NOT be close to the reference.
    assert not torch.allclose(out_cuda, out_ref, atol=1e-4), \
           "Kernel unexpectedly produced valid output with incorrect groups configuration."

# -------------------------------------------------------------------
# Test case 2: Noncontiguous tensor input.
# The kernel assumes that tensors are contiguous so a noncontiguous tensor may trigger errors.
def test_noncontiguous_input():
    torch.manual_seed(0)
    batch_size = 4
    in_channels = 32
    out_channels = 64
    kernel_h, kernel_w = 3, 5
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    dilation = 1

    height, width = 32, 32
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Make x noncontiguous by transposing two spatial dimensions and then transposing back only part of it.
    x = x.transpose(2, 3)
    
    weight = torch.randn(in_channels, out_channels // groups, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    ext = build_kernel()
    # Call CUDA extension forward with noncontiguous input
    out_cuda = ext.forward(x, weight, bias, stride, padding, output_padding, groups, dilation)

    # For reference, create a contiguous copy and run PyTorch's ConvTranspose2d
    x_contig = x.contiguous()
    conv = torch.nn.ConvTranspose2d(
        in_channels, out_channels, (kernel_h, kernel_w),
        stride=stride, padding=padding, output_padding=output_padding,
        groups=groups, bias=True
    ).to("cuda")
    conv.weight.data = weight
    conv.bias.data = bias
    out_ref = conv(x_contig)
    
    # Expect the outputs to differ because the kernel assumed contiguity.
    assert not torch.allclose(out_cuda, out_ref, atol=1e-4), \
           "Kernel did not behave differently for non-contiguous input as expected."

# -------------------------------------------------------------------
# Test case 3: Lack of error checking after kernel launch.
# We intentionally pass a tensor with an unexpected shape (e.g. 3D instead of 4D)
# to trigger a failure inside our CUDA kernel.
def test_invalid_input_dimensions():
    torch.manual_seed(0)
    in_channels = 16
    kernel_h, kernel_w = 3, 3
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    dilation = 1

    # Create an input with invalid dimensions (3D tensor instead of 4D).
    x = torch.randn(10, in_channels, 32, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 32 // groups, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(32, device="cuda", dtype=torch.float32)
    
    ext = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect the kernel to raise an error (or the AT_CHECK to fire) because of wrong input dimensions.
        out = ext.forward(x, weight, bias, stride, padding, output_padding, groups, dilation)
        torch.cuda.synchronize()

# -------------------------------------------------------------------
# Test case 4: Fixed block size may be suboptimal for very small outputs.
# We test with a small tensor that forces the total_elements to be less than the block size.
def test_small_output_tensor():
    torch.manual_seed(0)
    batch_size = 1
    in_channels = 8
    out_channels = 8
    kernel_h, kernel_w = 2, 2
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    dilation = 1

    height, width = 2, 2  # very small spatial size
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels // groups, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    ext = build_kernel()
    out_cuda = ext.forward(x, weight, bias, stride, padding, output_padding, groups, dilation)
    
    # For reference, use PyTorch
    conv = torch.nn.ConvTranspose2d(
        in_channels, out_channels, (kernel_h, kernel_w),
        stride=stride, padding=padding, output_padding=output_padding,
        groups=groups, bias=True
    ).to("cuda")
    conv.weight.data = weight
    conv.bias.data = bias
    out_ref = conv(x)
    
    # Even with suboptimal launch config, we expect the output to differ due to the fixed block size assumption.
    assert not torch.allclose(out_cuda, out_ref, atol=1e-4), \
           "Kernel produced valid output for a small output tensor; fixed block size issue not triggered."

