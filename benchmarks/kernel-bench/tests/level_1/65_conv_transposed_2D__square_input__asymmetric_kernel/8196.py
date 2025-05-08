
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility function to compile and load the CUDA kernel extension.
def build_kernel():
    cuda_module = load(
        name="conv_transpose2d_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper function to build a reference result using PyTorch's ConvTranspose2d.
def reference_conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups, dilation):
    conv = torch.nn.ConvTranspose2d(
        in_channels=x.size(1),
        out_channels=weight.size(0) * groups,  # conv_transpose2d expects weight shape: (in_channels, out_channels // groups, kernel_h, kernel_w)
        kernel_size=(weight.size(2), weight.size(3)),
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=(bias is not None)
    ).to(x.device)
    # Since our custom kernel expects weight to be in the shape (in_channels, out_channels//groups, kernel_h, kernel_w),
    # we copy our provided weight into conv.weight (transposing group blocks if necessary).
    with torch.no_grad():
        conv.weight.copy_(weight)
        if bias is not None:
            conv.bias.copy_(bias)
    return conv(x)

# Test case 1: Trigger the output_padding issue.
def test_output_padding_issue():
    # Set parameters such that output_padding > 0.
    batch_size = 4
    in_channels = 8
    out_channels = 16
    kernel_size = (3, 5)
    stride = 2
    padding = 1
    output_padding = 1  # Non-zero output_padding should affect the mapping.
    groups = 1
    dilation = 1

    # Create random input and weight tensors.
    x = torch.randn(batch_size, in_channels, 20, 20, device='cuda', dtype=torch.float32)
    # Weight for ConvTranspose2d: shape (in_channels, out_channels/groups, kernel_h, kernel_w)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1],
                         device='cuda', dtype=torch.float32)
    # Bias of shape (out_channels,) if used.
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    # Build the kernel.
    kernel_module = build_kernel()

    # Run the custom CUDA kernel.
    custom_out = kernel_module.forward(
        x, weight, bias, stride, padding, output_padding, groups, dilation
    )
    torch.cuda.synchronize()

    # Compute reference using PyTorch's ConvTranspose2d.
    ref_out = reference_conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups, dilation)
    torch.cuda.synchronize()

    # We expect the outputs to differ because the kernel ignores output_padding.
    # The test should detect a significant deviation.
    assert not torch.allclose(custom_out, ref_out, atol=1e-5), (
        "Custom kernel unexpectedly matches reference output when output_padding > 0. "
        "This indicates that the output_padding is being handled correctly, which is not expected."
    )

# Test case 2: Trigger the non-contiguous weight issue.
def test_non_contiguous_weight():
    # Use standard parameters.
    batch_size = 4
    in_channels = 8
    out_channels = 16
    kernel_size = (3, 3)
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    dilation = 1

    # Create random input.
    x = torch.randn(batch_size, in_channels, 15, 15, device='cuda', dtype=torch.float32)

    # Create weight tensor in the expected shape.
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1],
                         device='cuda', dtype=torch.float32)
    # Make the weight non-contiguous by transposing two dimensions and then transposing back.
    weight_non_contiguous = weight.transpose(0, 1).transpose(0, 1)
    # Verify non-contiguity.
    assert not weight_non_contiguous.is_contiguous(), "Weight tensor should be non-contiguous for this test."

    # Create bias.
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    # Build the kernel.
    kernel_module = build_kernel()

    # Run the custom CUDA kernel with a non-contiguous weight tensor.
    custom_out = kernel_module.forward(
        x, weight_non_contiguous, bias, stride, padding, output_padding, groups, dilation
    )
    torch.cuda.synchronize()

    # Compute reference using PyTorch's ConvTranspose2d.
    # Note: PyTorch will internally handle non-contiguous weights; if our custom kernel does not,
    # we expect a significant difference.
    ref_out = reference_conv_transpose2d(x, weight_non_contiguous, bias, stride, padding, output_padding, groups, dilation)
    torch.cuda.synchronize()

    # The outputs are likely to differ if the kernel incorrectly assumes contiguous memory.
    assert not torch.allclose(custom_out, ref_out, atol=1e-5), (
        "Custom kernel unexpectedly matches reference output with a non-contiguous weight tensor. "
        "This indicates that the kernel is not handling non-contiguous tensors as expected."
    )
                        
if __name__ == '__main__':
    pytest.main([__file__])
