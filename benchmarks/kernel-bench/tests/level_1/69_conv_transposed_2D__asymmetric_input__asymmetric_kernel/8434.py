
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger the groups issue.
# Note: When groups > 1, the weight layout should be (groups, C_in/groups, C_out/groups, kH, kW),
# but our kernel expects shape (C_in, C_out, kH, kW). This test uses grouped convolution parameters
# and compares against PyTorch's nn.ConvTranspose2d, expecting a mismatch.
def test_groups_issue():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    
    # Parameters for a grouped convolution.
    batch_size = 2
    in_channels = 4
    out_channels = 4
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    output_padding = (0, 0)
    dilation = (1, 1)
    groups = 2  # groups > 1

    # Create a ConvTranspose2d module to generate reference output.
    ref_conv_transpose = torch.nn.ConvTranspose2d(
        in_channels, 
        out_channels, 
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        bias=True
    ).cuda()
    
    # Extract tensors.
    x = torch.randn(batch_size, in_channels, 8, 8, device="cuda", dtype=torch.float32)
    # Note: PyTorch weight layout for grouped convolution stays as (in_channels, out_channels // groups, kH, kW)
    weight = ref_conv_transpose.weight.detach().clone()
    bias = ref_conv_transpose.bias.detach().clone() if ref_conv_transpose.bias is not None else None

    # Build kernel module.
    mod = build_kernel()

    # Using the custom kernel: it expects weight of shape (in_channels, out_channels, kH, kW)
    # So we deliberately reshape weight incorrectly by merging group dimension.
    # This will trigger wrong behavior.
    # For groups=2, valid weight should have shape: (4, 2, 3, 3). Instead, we create one with shape (4,4,3,3)
    weight_wrong = weight.repeat(1, groups, 1, 1)

    # Call the custom CUDA kernel
    output_custom = mod.forward(x, weight_wrong, bias, list(stride), list(padding), list(output_padding), list(dilation), groups)
    
    output_ref = ref_conv_transpose(x)
    
    # The outputs should not be close due to the ignored groups parameter.
    with pytest.raises(AssertionError):
        assert torch.allclose(output_custom, output_ref, atol=1e-5), "Grouped convolution issue not detected!"

# Test case 2: Trigger the data type issue.
# The kernel only supports float32. Passing a float64 tensor should lead to an error.
def test_dtype_issue():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    
    batch_size = 2
    in_channels = 8
    out_channels = 8
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    output_padding = (0, 0)
    dilation = (1, 1)
    groups = 1

    x = torch.randn(batch_size, in_channels, 8, 8, device="cuda", dtype=torch.float64)
    # Create weight and bias in float64.
    weight = torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float64)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float64)

    mod = build_kernel()

    with pytest.raises(RuntimeError):
        # The forward function in our kernel expects float32.
        _ = mod.forward(x, weight, bias, list(stride), list(padding), list(output_padding), list(dilation), groups)

# Test case 3: (Minor) Lack of kernel launch error checking.
# This test does not produce a wrong output per se, but we try to catch any asynchronous errors.
# We induce an error by using an invalid kernel parameter (e.g. negative padding)
def test_invalid_padding():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    
    batch_size = 2
    in_channels = 8
    out_channels = 8
    kernel_size = (3, 3)
    stride = (1, 1)
    # Introduce an invalid negative padding.
    padding = (-1, -1)
    output_padding = (0, 0)
    dilation = (1, 1)
    groups = 1

    x = torch.randn(batch_size, in_channels, 8, 8, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    mod = build_kernel()

    with pytest.raises(RuntimeError):
        # The kernel might launch but cause an asynchronous error which we try to catch.
        _ = mod.forward(x, weight, bias, list(stride), list(padding), list(output_padding), list(dilation), groups)
