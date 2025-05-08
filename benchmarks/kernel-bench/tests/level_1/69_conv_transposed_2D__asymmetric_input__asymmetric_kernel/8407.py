
import torch
import pytest
from torch.utils.cpp_extension import load

# Build the CUDA extension
def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper to compute reference output using PyTorch's ConvTranspose2d layer
def reference_conv_transpose2d(x, weight, bias, stride, padding, output_padding, dilation, groups):
    op = torch.nn.ConvTranspose2d(
        in_channels=x.size(1),
        out_channels=weight.size(1),
        kernel_size=(weight.size(2), weight.size(3)),
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        bias=bias is not None,
    ).to(x.device).type_as(x)
    
    # Set weights and bias to be identical.
    with torch.no_grad():
        op.weight.copy_(weight)
        if bias is not None:
            op.bias.copy_(bias)
    return op(x)

# Test case 1: Trigger the reduction issue.
# Use a configuration that forces more than one thread to process the same output pixel.
# We use a number of input channels greater than SMALL_CHANNEL_THRESHOLD (64) so that
# blocks_per_out > 1 and multiple threads compute partial sums.
def test_intra_block_reduction_issue():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 128  # > SMALL_CHANNEL_THRESHOLD (64) forces custom kernel
    out_channels = 32
    input_height, input_width = 8, 8
    kernel_size = (3, 3)  # small kernel but in_channels forces custom kernel
    stride = (1, 1)
    padding = (1, 1)
    output_padding = (0, 0)
    dilation = (1, 1)
    groups = 1
    bias_flag = True

    x = torch.randn(batch_size, in_channels, input_height, input_width, device='cuda', dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1], device='cuda', dtype=torch.float32)
    bias_tensor = torch.randn(out_channels, device='cuda', dtype=torch.float32) if bias_flag else None

    # Get output from custom CUDA kernel.
    out_custom = cuda_module.forward(
        x, weight, bias_tensor, list(stride), list(padding), list(output_padding), list(dilation), groups
    )

    # Get reference output from PyTorch.
    out_ref = reference_conv_transpose2d(x, weight, bias_tensor, stride, padding, output_padding, dilation, groups)

    # The custom kernel is expected to have wrong reductions, so outputs should differ.
    assert not torch.allclose(out_custom, out_ref, atol=1e-3), \
        "Test failed: Custom kernel output unexpectedly matches reference output despite missing reduction."

# Test case 2: Trigger the dilation issue.
# Use a dilation value different from (1, 1). Since the kernel ignores dilation, the result will be wrong.
def test_dilation_issue():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 32
    out_channels = 16
    input_height, input_width = 8, 8
    kernel_size = (7, 7)  # Use kernel size greater than SMALL_KERNEL_THRESHOLD so that custom kernel is used.
    stride = (1, 1)
    padding = (3, 3)
    output_padding = (0, 0)
    dilation = (2, 2)  # Non-trivial dilation, but kernel ignores it.
    groups = 1
    bias_flag = False

    x = torch.randn(batch_size, in_channels, input_height, input_width, device='cuda', dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1], device='cuda', dtype=torch.float32)
    bias_tensor = None

    # Custom kernel forward.
    out_custom = cuda_module.forward(
        x, weight, bias_tensor, list(stride), list(padding), list(output_padding), list(dilation), groups
    )

    # Reference output using proper dilation.
    out_ref = reference_conv_transpose2d(x, weight, bias_tensor, stride, padding, output_padding, dilation, groups)

    # Since dilation is not used in custom kernel, outputs should not match.
    assert not torch.allclose(out_custom, out_ref, atol=1e-3), \
        "Test failed: Custom kernel output unexpectedly matches reference output despite incorrect handling of dilation."

# Test case 3: Trigger the groups issue.
# Use groups > 1 so that the convolution becomes a grouped convolution.
def test_groups_issue():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 16
    out_channels = 16
    input_height, input_width = 8, 8
    kernel_size = (7, 7)  # Use a kernel size that forces the custom kernel path.
    stride = (1, 1)
    padding = (3, 3)
    output_padding = (0, 0)
    dilation = (1, 1)
    groups = 2  # groups > 1
    bias_flag = True

    x = torch.randn(batch_size, in_channels, input_height, input_width, device='cuda', dtype=torch.float32)
    # For grouped convolution, weight shape is (in_channels, out_channels//groups, kH, kW)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], device='cuda', dtype=torch.float32)
    bias_tensor = torch.randn(out_channels, device='cuda', dtype=torch.float32) if bias_flag else None

    # Custom kernel forward.
    out_custom = cuda_module.forward(
        x, weight, bias_tensor, list(stride), list(padding), list(output_padding), list(dilation), groups
    )

    # Reference output using PyTorch.
    out_ref = reference_conv_transpose2d(x, weight, bias_tensor, stride, padding, output_padding, dilation, groups)

    # The kernel does not support groups, so the output should differ.
    assert not torch.allclose(out_custom, out_ref, atol=1e-3), \
        "Test failed: Custom kernel output unexpectedly matches reference output despite ignoring groups parameter."
