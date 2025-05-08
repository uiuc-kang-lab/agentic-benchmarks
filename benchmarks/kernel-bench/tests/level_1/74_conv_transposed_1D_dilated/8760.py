
import pytest
import torch
from torch.utils.cpp_extension import load
import torch.nn.functional as F

# Utility function to build and load the CUDA extension module.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Data type handling.
# The kernel expects float (float32) tensors.
def test_dtype_support():
    kernel_mod = build_kernel()
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = 3
    length = 10
    stride = 1
    padding = 0
    dilation = 1

    # Create input, weight, and bias tensors in double (float64)
    x = torch.randn(batch_size, in_channels, length, dtype=torch.double, device="cuda")
    # For nn.ConvTranspose1d in PyTorch, the weight shape is (in_channels, out_channels, kernel_size)
    weight = torch.randn(in_channels, out_channels, kernel_size, dtype=torch.double, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.double, device="cuda")
    # We expect that calling the CUDA op on a double tensor leads to undefined behavior
    # so our test should detect that the results are not matching a reference.
    # Here we simply compare against PyTorch’s own conv_transpose1d.
    with pytest.raises(RuntimeError):
        # This call will use data_ptr<float>() on a double tensor which should trigger an error.
        kernel_mod.forward(x, weight, bias, stride, padding, dilation)

# Issue 2: Grouped convolutions.
# The kernel assumes weight shape corresponding to groups=1.
def test_group_convolution():
    kernel_mod = build_kernel()
    # Setup a grouped convolution case:
    batch_size = 1
    in_channels = 4     # Must be divisible by groups.
    out_channels = 8
    groups = 2
    kernel_size = 3
    length = 10
    stride = 1
    padding = 0
    dilation = 1

    # PyTorch's nn.ConvTranspose1d with groups != 1 will have weight shape of
    # (in_channels, out_channels // groups, kernel_size)
    # Our kernel, however, expects weight shape: [in_channels, out_channels, kernel_size]
    # Build input and weight for the grouped conv.
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    # Create weight with grouped layout
    weight_grouped = torch.randn(in_channels, out_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    bias_grouped = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    # Compute conv_transpose1d using PyTorch's native grouped layer.
    conv_native = torch.nn.ConvTranspose1d(
        in_channels, out_channels, kernel_size, stride=stride,
        padding=padding, dilation=dilation, groups=groups, bias=True
    ).to(device="cuda", dtype=torch.float32)
    # Override weight and bias with our grouped tensors.
    # Note: For a grouped conv, PyTorch expects weight shape [in_channels, out_channels // groups, kernel_size]
    with torch.no_grad():
        conv_native.weight.copy_(weight_grouped)
        conv_native.bias.copy_(bias_grouped)
    
    y_native = conv_native(x)
    
    # Now call the custom kernel.
    # Since the custom kernel does not support groups, we simulate what happens if we call it:
    # We must pass a weight tensor of shape [in_channels, out_channels, kernel_size].
    # Here we “pad” the weight along the second dimension by repeating elements,
    # which is an incorrect behavior. We do that to force the kernel into action.
    weight_wrong = weight_grouped.repeat(1, groups, 1)
    y_kernel = kernel_mod.forward(x, weight_wrong, bias_grouped, stride, padding, dilation)
    
    # The expected output channels for the native conv is out_channels,
    # but our kernel will compute with out_channels inferred as weight_wrong.size(1) = out_channels (okay),
    # however the convolution result will be computed incorrectly because channel grouping is not taken into account.
    # Therefore, check that the results are different.
    assert not torch.allclose(y_kernel, y_native, atol=1e-4), "Kernel did not expose the grouping error!"

# Issue 3: output_padding support.
# The kernel does not account for an output_padding parameter,
# which is necessary for some transposed convolution configurations.
def test_output_padding():
    kernel_mod = build_kernel()
    batch_size = 1
    in_channels = 3
    out_channels = 5
    kernel_size = 3
    length = 8
    stride = 2
    padding = 1
    dilation = 1
    output_padding = 1  # nonzero output padding

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    # Compute output shape as per PyTorch's formula with output_padding:
    # L_out_native = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    L_out_native = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

    # Use PyTorch's native conv_transpose1d with output_padding.
    conv_native = torch.nn.ConvTranspose1d(
        in_channels, out_channels, kernel_size, stride=stride,
        padding=padding, output_padding=output_padding, dilation=dilation, bias=True
    ).to(device="cuda", dtype=torch.float32)
    with torch.no_grad():
        conv_native.weight.copy_(weight)
        conv_native.bias.copy_(bias)
    y_native = conv_native(x)
    assert y_native.size(2) == L_out_native, "Native conv_transpose1d output length mismatch."

    # The custom kernel computes L_out without output_padding:
    L_out_kernel = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    y_kernel = kernel_mod.forward(x, weight, bias, stride, padding, dilation)
    
    # Check that output length from kernel is not equal to the expected length when output_padding is nonzero.
    assert y_kernel.size(2) == L_out_kernel, "Kernel computed L_out incorrectly even without output_padding."
    assert y_kernel.size(2) != L_out_native, "Kernel unexpectedly supports output_padding!"
