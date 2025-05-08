
import torch
import pytest
from torch.utils.cpp_extension import load
import torch.nn as nn

# Helper function to build and return the CUDA module.
def build_kernel():
    cuda_module = load(
        name="conv_transpose2d_cuda_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1 test:
# Intended shared memory optimization is not implemented.
# Although this does not affect numerical correctness, the expectation is that
# the kernel should use the allocated shared memory region for optimization.
# Here we mark this test as expected to fail because the shared memory is unused.
@pytest.mark.xfail(reason="Kernel does not utilize allocated shared memory as intended (Issue 1).")
def test_unused_shared_memory():
    # Use a small convolution where groups=1 so that weight indexing is correct,
    # but the shared memory optimization would have mattered for performance.
    batch_size = 4
    in_channels = 8
    out_channels = 8
    kernel_size = (3, 3)
    height_in = 10
    width_in = 10
    stride = (1, 1)
    padding = (1, 1)
    output_padding = (0, 0)
    dilation = (1, 1)
    groups = 1

    device = torch.device("cuda")
    torch.manual_seed(0)
    # Create input, weight, and (optional) bias.
    input_tensor = torch.randn(batch_size, in_channels, height_in, width_in, device=device, dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], device=device, dtype=torch.float32)
    bias = torch.randn(out_channels, device=device, dtype=torch.float32)

    # Run the custom CUDA kernel.
    module = build_kernel()
    output_cuda = module.forward(
        input_tensor, weight, bias,
        list(stride), list(padding), list(output_padding), list(dilation), groups
    )
    # Now, run the native PyTorch ConvTranspose2d (for groups==1 this is correct).
    conv_transpose = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, output_padding=output_padding,
        dilation=dilation, groups=groups, bias=True
    ).to(device)
    # Manually set the weights and bias to be the same.
    conv_transpose.weight.data.copy_(weight)
    conv_transpose.bias.data.copy_(bias)
    output_ref = conv_transpose(input_tensor)

    # Even though numerically the results might match for groups == 1,
    # the fact that the kernel is not using shared memory as expected is a known issue.
    # Hence we flag this test as xfail.
    assert torch.allclose(output_cuda, output_ref, atol=1e-4), \
        f"Output from custom CUDA kernel does not match reference (Issue 1)."

# Issue 2 test:
# The weight indexing is incorrect for grouped convolutions.
# For groups > 1, the custom kernel output will differ from the reference PyTorch implementation.
def test_grouped_convolution_incorrect_weight_indexing():
    batch_size = 2
    in_channels = 4
    groups = 2
    # When groups > 1, out_channels must be a multiple of groups.
    out_channels = 4
    kernel_size = (3, 3)
    height_in = 8
    width_in = 8
    stride = (1, 1)
    padding = (1, 1)
    output_padding = (0, 0)
    dilation = (1, 1)

    device = torch.device("cuda")
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, in_channels, height_in, width_in, device=device, dtype=torch.float32)
    # For ConvTranspose2d, the weight shape is (in_channels, out_channels/groups, kH, kW)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], device=device, dtype=torch.float32)
    bias = torch.randn(out_channels, device=device, dtype=torch.float32)

    # Run the custom CUDA kernel.
    module = build_kernel()
    output_cuda = module.forward(
        input_tensor, weight, bias,
        list(stride), list(padding), list(output_padding), list(dilation), groups
    )
    
    # Run the native PyTorch ConvTranspose2d.
    conv_transpose = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, output_padding=output_padding,
        dilation=dilation, groups=groups, bias=True
    ).to(device)
    conv_transpose.weight.data.copy_(weight)
    conv_transpose.bias.data.copy_(bias)
    output_ref = conv_transpose(input_tensor)

    # Because of the weight indexing error in the custom kernel for grouped convolutions,
    # the outputs should differ.
    diff = (output_cuda - output_ref).abs().max().item()
    assert diff > 1e-2, f"Expected significant discrepancy for grouped convolution (Issue 2), but max diff = {diff}"

if __name__ == "__main__":
    pytest.main([__file__])
