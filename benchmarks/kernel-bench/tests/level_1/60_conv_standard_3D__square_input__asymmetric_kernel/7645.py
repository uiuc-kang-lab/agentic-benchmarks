
import torch
import pytest
from torch import nn
from torch.utils.cpp_extension import load

# Build the CUDA kernel module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger bias descriptor issue.
# When a bias tensor is provided, the kernel should add bias correctly.
# Since the bias descriptor is not properly initialized, the output will differ from the PyTorch Conv3d result.
def test_bias_descriptor_issue():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    batch_size = 4
    in_channels = 3
    out_channels = 8
    # Use an asymmetric kernel
    kernel_size = (3, 5, 7)
    stride = 1
    padding = 1
    dilation = 1
    groups = 1
    bias_flag = True

    # Create PyTorch Conv3d layer with bias
    conv = nn.Conv3d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias_flag
    ).cuda().eval()
    
    # Generate random input
    x = torch.randn(batch_size, in_channels, 16, 16, 16, device="cuda")
    with torch.no_grad():
        ref_output = conv(x)
    
    # Extract weights and bias from the PyTorch module
    weight = conv.weight.detach()
    bias = conv.bias.detach()
    
    # Load the custom kernel module
    kernel_module = build_kernel()
    
    # Call the custom CUDA kernel forward function
    try:
        custom_out = kernel_module.forward(
            x, weight, bias, stride, padding, dilation, groups
        )
    except Exception as e:
        pytest.skip(f"Kernel threw an exception (possibly due to bias descriptor): {e}")
    
    # The outputs should match if bias is correctly processed.
    # Here we expect a significant difference due to the uninitialized bias descriptor.
    diff = (custom_out - ref_output).abs().max().item()
    assert diff > 1e-3, (
        f"Bias descriptor issue may not have been triggered. Max difference: {diff}"
    )

# Test case 2: Trigger alpha/beta data type issue.
# When the inputs are not of type float (e.g. double), the hard-coded float alpha/beta
# can lead to precision/type mismatches, causing output differences.
def test_alpha_beta_type_issue():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    batch_size = 4
    in_channels = 3
    out_channels = 8
    kernel_size = (3, 3, 3)
    stride = 1
    padding = 1
    dilation = 1
    groups = 1
    bias_flag = False

    # Create PyTorch Conv3d layer with bias disabled, using double precision.
    conv = nn.Conv3d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias_flag
    ).double().cuda().eval()

    x = torch.randn(batch_size, in_channels, 16, 16, 16, device="cuda", dtype=torch.double)
    with torch.no_grad():
        ref_output = conv(x)

    weight = conv.weight.detach()
    
    kernel_module = build_kernel()

    try:
        custom_out = kernel_module.forward(
            x, weight, None, stride, padding, dilation, groups
        )
    except Exception as e:
        pytest.skip(f"Kernel threw an exception (possibly due to type mismatch): {e}")
    
    diff = (custom_out - ref_output).abs().max().item()
    # Expect a significant difference due to wrong alpha/beta type usage
    assert diff > 1e-3, (
        f"Alpha/Beta type issue may not have been triggered. Max difference: {diff}"
    )

# Test case 3: Trigger error checking for non-CUDA input.
# Passing a CPU tensor should trigger a TORCH_CHECK error.
def test_non_cuda_input():
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = (3, 3, 3)
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    # Create CPU tensors intentionally
    x = torch.randn(batch_size, in_channels, 16, 16, 16, device="cpu")
    weight = torch.randn(out_channels, in_channels // groups, *kernel_size, device="cpu")
    
    kernel_module = build_kernel()
    
    with pytest.raises(RuntimeError, match="Input must be a CUDA tensor"):
        kernel_module.forward(x, weight, None, stride, padding, dilation, groups)
