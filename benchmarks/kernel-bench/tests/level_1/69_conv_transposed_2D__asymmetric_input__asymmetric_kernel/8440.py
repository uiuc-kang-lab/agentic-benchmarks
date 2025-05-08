
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# Build the CUDA kernel extension from kernel.cu
def build_kernel():
    return load(
        name="transposed_conv_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# 1. Test case for groups != 1
def test_groups_not_one():
    # Create input and weight for a grouped convolution (groups > 1)
    # Our kernel assumes groups==1 so the output will be wrong.
    batch_size = 2
    in_channels = 4
    out_channels = 4  # must be divisible by groups
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    output_padding = (0, 0)
    dilation = (1, 1)
    groups = 2  # non-default

    # Use PyTorch's built-in module for a reference
    ref_module = nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        bias=True,
    ).cuda()
    # Use built kernel module even though it cannot support groups>1 properly.
    # We expect the result to differ from the reference.
    x = torch.randn(batch_size, in_channels, 8, 8, device="cuda", dtype=torch.float32)
    kernel_module = build_kernel()

    # Prepare weight and bias in the same layout as assumed by the kernel
    # The kernel assumes weight shape: [in_channels, out_channels, kh, kw] and groups==1.
    weight = ref_module.weight.detach().clone()
    bias = ref_module.bias.detach().clone()
    # In grouped convolution, the weight layout is different.
    # Therefore, we expect a discrepancy.
    out_kernel = kernel_module.forward(x, weight, bias, list(stride), list(padding), list(output_padding), list(dilation), groups)
    out_ref = ref_module(x)
    
    # The two outputs must be different since kernel implementation was not modified to support groups.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), "Kernel unexpectedly worked for groups!=1"

# 2. Test case for non-float32 tensor type
def test_non_float32_input():
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    output_padding = (0, 0)
    dilation = (1, 1)
    groups = 1

    # Create the reference module which expects float32 weights etc.
    module = nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        bias=True,
    ).cuda()
    
    # Create double precision input and weight. The kernel expects float32.
    x = torch.randn(batch_size, in_channels, 8, 8, device="cuda", dtype=torch.float64)
    weight = module.weight.detach().clone().to(torch.float64)
    bias = module.bias.detach().clone().to(torch.float64)
    
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Casting is not done automatically in our kernel.
        kernel_module.forward(x, weight, bias, list(stride), list(padding), list(output_padding), list(dilation), groups)

# 3. Test case for noncontiguous input tensor
def test_non_contiguous_input():
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)
    output_padding = (1, 1)
    dilation = (1, 1)
    groups = 1

    module = nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        bias=True,
    ).cuda()
    
    x = torch.randn(batch_size, in_channels, 8, 8, device="cuda", dtype=torch.float32)
    # Make input noncontiguous by transposing two dimensions and then undoing
    x_noncontig = x.transpose(2, 3)
    if x_noncontig.is_contiguous():
        pytest.skip("Test requires a noncontiguous tensor, but got a contiguous one.")
    
    kernel_module = build_kernel()
    # Even though torch.nn.ConvTranspose2d can handle noncontiguous inputs,
    # our kernel does not check for it and relies on data_ptr access.
    out_kernel = kernel_module.forward(x_noncontig, module.weight.detach().clone(), module.bias.detach().clone(), list(stride), list(padding), list(output_padding), list(dilation), groups)
    # Compare with PyTorch conv-transpose on the noncontiguous tensor (PyTorch will handle it correctly internally).
    out_ref = module(x_noncontig.contiguous())
    # Likely the result will differ due to the assumption of contiguity in the kernel.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), "Kernel unexpectedly worked on a noncontiguous input"

# 4. Test case for branchless mod computation on negative numbers
def test_negative_modulo_behavior():
    # This test creates a situation where the computed h_diff and w_diff become negative.
    # For example, using padding=0 and an output location near 0.
    batch_size = 1
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (0, 0)
    output_padding = (0, 0)
    dilation = (1, 1)
    groups = 1

    module = nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        bias=False,
    ).cuda()
    
    # Create an input such that for output pixel (0,0) h_diff and w_diff are negative.
    x = torch.randn(batch_size, in_channels, 2, 2, device="cuda", dtype=torch.float32)
    kernel_module = build_kernel()
    out_kernel = kernel_module.forward(x, module.weight.detach().clone(), None, list(stride), list(padding), list(output_padding), list(dilation), groups)
    out_ref = module(x)
    
    # We expect a discrepancy (or error) because the modulo computations might produce different valid mask values.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), "Kernel's branchless arithmetic did not trigger an error with negative differences"

if __name__ == "__main__":
    pytest.main([__file__])
