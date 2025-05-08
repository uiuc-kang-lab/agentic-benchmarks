
import torch
import pytest
from torch import nn
from torch.utils.cpp_extension import load

# Utility function to build the CUDA extension module (assumes kernel.cu is in the current directory)
def build_kernel():
    cuda_module = load(
        name="conv_transpose2d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper: Custom forward calling our CUDA kernel forward function.
def custom_conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, module):
    # the custom kernel forward function signature is:
    # conv_transpose2d_forward(x, weight, bias (or None), stride, padding, output_padding, groups)
    return module.forward(input, weight, bias, stride, padding, output_padding, groups)

# Test case 1:
# Test for non-square kernel issue.
# We build a weight tensor with non-square dimensions (kernel height != kernel width)
# Our kernel only accepts one "kernel_size" (from weight.size(2)) and then uses that for both dims, so this should trigger an issue.
def test_non_square_kernel():
    cuda_mod = build_kernel()
    batch = 2
    in_channels = 4
    out_channels = 8
    height_in = 16
    width_in = 16
    # Create non-square weight. Expected weight shape for conv_transpose2d is [in_channels, out_channels//groups, kH, kW]
    kH, kW = 3, 5   # non-square kernel
    weight = torch.randn(in_channels, out_channels, kH, kW, device="cuda", dtype=torch.float32)
    bias = None
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    x = torch.randn(batch, in_channels, height_in, width_in, device="cuda", dtype=torch.float32)
    # Since our custom kernel will extract kernel_size from weight.size(2) (which is 3) and will use that value for both dims,
    # while the weight tensor has an actual width of 5, the indexing will be off.
    # We expect the output of the custom kernel to differ significantly from PyTorch's built-in conv_transpose2d.
    custom_out = custom_conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups, cuda_mod)
    # Build a reference layer using nn.ConvTranspose2d with non-square kernel;
    # PyTorch supports non-square kernels if passed as a tuple.
    ref_conv = nn.ConvTranspose2d(
        in_channels, out_channels,
        kernel_size=(kH, kW),
        stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=False
    ).cuda()
    # Replace the reference conv weights with our custom weight, but note that our weight shape ordering is different.
    # PyTorch's weight shape for ConvTranspose2d is (in_channels, out_channels//groups, kH, kW).
    # Our custom weight has shape (in_channels, out_channels, kH, kW). Split the second dimension:
    ref_conv.weight.data = weight.view(in_channels, out_channels//groups, kH, kW)
    ref_out = ref_conv(x)
    # Because the custom kernel assumes a square kernel based on kH, the outputs will not match.
    with pytest.raises(AssertionError):
        assert torch.allclose(custom_out, ref_out, atol=1e-4), "Non-square kernel: Custom kernel produced output similar to reference!"

# Test case 2:
# Test for unsupported dilation.
# Since the custom kernel does not take dilation into account, its output will differ when dilation != 1.
def test_unsupported_dilation():
    cuda_mod = build_kernel()
    batch = 2
    in_channels = 4
    out_channels = 8
    height_in = 16
    width_in = 16
    kernel_size = 3
    dilation = 2  # non-unit dilation
    bias = None
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    x = torch.randn(batch, in_channels, height_in, width_in, device="cuda", dtype=torch.float32)
    # Create weight with standard shape. The custom kernel ignores dilation so weight shape must be square.
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)

    # Use built-in transposed convolution with dilation:
    ref_conv = nn.ConvTranspose2d(
        in_channels, out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        output_padding=output_padding, groups=groups, bias=False, dilation=dilation
    ).cuda()
    # Override weights for fair comparison
    ref_conv.weight.data = weight.view(in_channels, out_channels//groups, kernel_size, kernel_size)
    ref_out = ref_conv(x)

    # Custom kernel call: note that it has no dilation parameter and effectively assumes dilation=1.
    custom_out = custom_conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups, cuda_mod)

    # Because dilation is not handled, the outputs will differ.
    with pytest.raises(AssertionError):
        assert torch.allclose(custom_out, ref_out, atol=1e-4), "Dilation issue: Custom kernel unexpectedly matched reference with dilation!"

# Test case 3:
# Test for edge-case when stride is larger than kernel dimensions.
# In such a configuration the modulus adjustments for loop bounds may yield incorrect results.
def test_high_stride_edge():
    cuda_mod = build_kernel()
    batch = 1
    in_channels = 2
    out_channels = 4
    height_in = 8
    width_in = 8
    kernel_size = 3
    stride = 4  # high stride compared to kernel size
    padding = 1
    output_padding = 0
    groups = 1
    bias = None

    x = torch.randn(batch, in_channels, height_in, width_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)

    # Built-in conv_transpose2d reference layer using high stride.
    ref_conv = nn.ConvTranspose2d(
        in_channels, out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        output_padding=output_padding, groups=groups, bias=False
    ).cuda()
    ref_conv.weight.data = weight.view(in_channels, out_channels//groups, kernel_size, kernel_size)
    ref_out = ref_conv(x)

    custom_out = custom_conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups, cuda_mod)
    
    # Expect the outputs to differ due to edge-case miscalculations in loop bounds.
    with pytest.raises(AssertionError):
        assert torch.allclose(custom_out, ref_out, atol=1e-4), "High-stride edge-case: Custom kernel output unexpectedly matches reference!"

# Note: The unused variable "out_ch_offset" is a code quality/maintenance issue rather than a runtime error,
# so there is no direct runtime test to trigger it.

if __name__ == "__main__":
    pytest.main([__file__])
