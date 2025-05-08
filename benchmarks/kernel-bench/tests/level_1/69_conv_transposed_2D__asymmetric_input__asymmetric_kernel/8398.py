
import torch
import pytest
from torch.utils.cpp_extension import load
import numpy as np

# Helper function to compile and load the CUDA kernel module
def build_kernel():
    module = load(
        name="test_conv_transpose2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Test case 1: Groups Issue
#
# When groups > 1 the kernel indexing for weight is wrong, so its output
# will differ from the reference PyTorch implementation.
def test_groups_issue():
    torch.manual_seed(0)
    device = "cuda"
    batch_size = 2
    in_channels = 4  # Must be divisible by groups
    out_channels = 4
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    output_padding = (0, 0)
    dilation = (1, 1)
    groups = 2  # groups > 1 to trigger the issue

    # Create input and weight tensors.
    x = torch.randn(batch_size, in_channels, 8, 8, device=device, dtype=torch.float32)
    # PyTorch's ConvTranspose2d for groups uses weight shape [in_channels, out_channels//groups, kH, kW]
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1],
                         device=device, dtype=torch.float32)
    bias = None

    # Standard output using PyTorch
    conv_tr = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding,
                                        output_padding=output_padding,
                                        dilation=dilation, groups=groups, bias=False).to(device)
    # Force the same weight in the pytorch model:
    with torch.no_grad():
        conv_tr.weight.copy_(weight)
    ref = conv_tr(x)

    # Custom kernel's host function is expecting weight in shape: [in_channels, out_channels, kH, kW]
    # so here we simulate the situation by expanding the weight to this shape incorrectly:
    # (the kernel ignores groups and expects full connectivity).
    # We'll simply use a weight tensor with shape [in_channels, out_channels, kH, kW]
    expanded_weight = weight.repeat(1, groups, 1, 1)  # replicate channels to mimic wrong layout

    my_cuda = build_kernel()
    # Call the custom function. Note: dilation and groups are passed from the CPU side but ignored in the kernel.
    out_custom = my_cuda.forward(x, expanded_weight, torch.tensor([]), 
                                 list(stride), list(padding), list(output_padding),
                                 list(dilation), groups)
    torch.cuda.synchronize()

    # Since the kernel implementation does not support groups, the output will be different.
    assert not torch.allclose(out_custom, ref, atol=1e-5), \
        "Test failed: Custom kernel output matches reference output unexpectedly for groups > 1."

# Test case 2: Dilation Issue
#
# When dilation > 1 the kernel ignores this argument, so the output should differ
# from the expected transposed convolution reference.
def test_dilation_issue():
    torch.manual_seed(0)
    device = "cuda"
    batch_size = 2
    in_channels = 3
    out_channels = 5
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)
    output_padding = (1, 1)
    dilation = (2, 2)  # dilation > 1
    groups = 1

    x = torch.randn(batch_size, in_channels, 10, 10, device=device, dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1],
                         device=device, dtype=torch.float32)
    bias = None

    conv_tr = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding,
                                        output_padding=output_padding,
                                        dilation=dilation, groups=groups, bias=False).to(device)
    with torch.no_grad():
        conv_tr.weight.copy_(weight)
    ref = conv_tr(x)

    my_cuda = build_kernel()
    out_custom = my_cuda.forward(x, weight, torch.tensor([]),
                                 list(stride), list(padding), list(output_padding),
                                 list(dilation), groups)
    torch.cuda.synchronize()

    # Since the kernel ignores dilation, its output will be notably different from the reference.
    assert not torch.allclose(out_custom, ref, atol=1e-5), \
        "Test failed: Custom kernel output unexpectedly matches reference output when dilation > 1."

# Test case 3: Grid Launch Configuration Issue (Large Output)
#
# Using a very large output size (hence many threads) may trigger the grid limitation
# (limited to 65535 blocks in a 1D grid), causing incorrect results.
def test_large_output_issue():
    torch.manual_seed(0)
    device = "cuda"
    # Create a scenario with very large output dimensions.
    batch_size = 1
    in_channels = 8
    out_channels = 16
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)
    output_padding = (1, 1)
    dilation = (1, 1)
    groups = 1

    # Choose an input size that creates huge outputs.
    # Example: input spatial size = 256, with stride=2 -> output around 513.
    x = torch.randn(batch_size, in_channels, 256, 256, device=device, dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1],
                         device=device, dtype=torch.float32)
    bias = None

    conv_tr = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding,
                                        output_padding=output_padding,
                                        dilation=dilation, groups=groups, bias=False).to(device)
    with torch.no_grad():
        conv_tr.weight.copy_(weight)
    ref = conv_tr(x)

    my_cuda = build_kernel()
    # This custom kernel uses a 1D grid limited to 65535 blocks.
    out_custom = my_cuda.forward(x, weight, torch.tensor([]),
                                 list(stride), list(padding), list(output_padding),
                                 list(dilation), groups)
    torch.cuda.synchronize()

    # For very large outputs, the custom kernel may not cover all elements correctly.
    # Therefore, we expect a significant mismatch between the custom output and reference.
    diff = (out_custom - ref).abs().max().item()
    assert diff > 1e-3, f"Test failed: The kernel appears to handle large outputs correctly (max diff {diff})."

if __name__ == "__main__":
    pytest.main([__file__])
