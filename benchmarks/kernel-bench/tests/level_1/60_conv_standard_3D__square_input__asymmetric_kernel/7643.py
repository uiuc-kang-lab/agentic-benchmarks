
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_conv3d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger type mismatch by using double precision tensors.
def test_double_precision_mismatch():
    # Create a double precision input and weight for 3D convolution.
    # The kernel uses hard-coded float alpha and beta, so this should trigger an error or produce wrong results.
    batch_size = 2
    in_channels = 3
    out_channels = 4
    # Use a cubic volume for simplicity.
    D = H = W = 16
    kernel_d, kernel_h, kernel_w = 3, 3, 3

    # Create tensors in double precision (torch.float64)
    input = torch.randn(batch_size, in_channels, D, H, W, dtype=torch.float64, device="cuda")
    weight = torch.randn(out_channels, in_channels, kernel_d, kernel_h, kernel_w, dtype=torch.float64, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float64, device="cuda")
    
    cuda_module = build_kernel()
    
    with pytest.raises(Exception) as excinfo:
        # This call is expected to fail because the alpha/beta values are float and the tensors are double.
        cuda_module.forward(input, weight, bias, 1, 0, 1, 1)
    assert "Unsupported" in str(excinfo.value) or "Error" in str(excinfo.value)

# Test case 2: Trigger dimension ordering issue.
def test_dimension_ordering_issue():
    # Here we deliberately create an input tensor with dimensions corresponding to
    # (batch, channels, width, height, depth) as per the Model’s documentation,
    # while the kernel interprets input as (batch, channels, depth, height, width).
    batch_size = 2
    in_channels = 3
    out_channels = 4
    # Choose asymmetric dimensions where width != depth.
    width = 20   # documented as 'width'
    height = 30  # documented as 'height'
    depth = 40   # documented as 'depth'
    kernel_size = (3, 5, 7)  # (kernel_d, kernel_h, kernel_w)

    # Build input tensor with ordering as (N, C, width, height, depth)
    input = torch.randn(batch_size, in_channels, width, height, depth, dtype=torch.float32, device="cuda")
    # Build weight tensor as used by nn.Conv3d which interprets kernel dims as (depth, height, width).
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2],
                         dtype=torch.float32, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")

    cuda_module = build_kernel()
    
    # Perform convolution using the custom kernel.
    output_custom = cuda_module.forward(input, weight, bias, 1, 0, 1, 1)
    
    # Build a standard nn.Conv3d module.
    # Note: nn.Conv3d expects input shape (N, C, D, H, W). If the user mistakenly passes (N, C, width, height, depth),
    # the built-in module will interpret width as depth. This test compares the output shapes.
    conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True).to("cuda")
    # Manually assign weight and bias to the nn.Conv3d module so that both convolutions use the same parameters.
    conv.weight.data.copy_(weight)
    conv.bias.data.copy_(bias)
    
    # For the test, we reinterpret the input by permuting dimensions to match nn.Conv3d expected order:
    # from (N, C, width, height, depth) to (N, C, depth, height, width)
    input_reordered = input.permute(0, 1, 4, 3, 2).contiguous()
    output_pytorch = conv(input_reordered)
    
    # If the kernel assumed the wrong dimension ordering, output_custom shape will not match output_pytorch shape.
    assert output_custom.shape != output_pytorch.shape, (
        "Expected a dimension mismatch due to ordering issue, but the output shapes are identical."
    )
