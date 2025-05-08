
import torch
import torch.nn.functional as F
import pytest
from torch.utils.cpp_extension import load

# Helper function to compile and load the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="custom_depthwise_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Groups parameter issue.
# Here we deliberately use groups=1 (standard convolution) with an input tensor that has in_channels > 1.
# The custom CUDA kernel ignores the groups parameter and computes a depthwise conv,
# so its output will differ from torch.nn.functional.conv2d used with groups=1.
def test_groups_parameter_issue():
    device = "cuda"
    batch_size = 2
    in_channels = 3
    height = 16
    width = 16
    kernel_size = 3
    stride = 1
    padding = 1
    
    # Standard conv2d weights for groups=1 have shape (out_channels, in_channels, k, k).
    # Here we use out_channels = in_channels for a fair comparison.
    weight = torch.randn(in_channels, in_channels, kernel_size, kernel_size, device=device, dtype=torch.float32)
    bias = torch.randn(in_channels, device=device, dtype=torch.float32)
    x = torch.randn(batch_size, in_channels, height, width, device=device, dtype=torch.float32)
    
    # Using the custom kernel which is written for depthwise conv.
    # It expects weight of shape (in_channels, 1, k, k) but we deliberately supply a weight for groups=1.
    custom_weight = weight.clone()  # This weight shape does not match what the kernel expects.
    
    # We use the forward_wrap from the extension.
    mod = build_kernel()
    # Call custom kernel with groups=1. Because our kernel ignores groups, it will index weight incorrectly.
    out_custom = mod.forward(x, custom_weight, bias, stride, padding, groups=1)
    
    # Reference standard convolution (groups=1) using PyTorch built-in conv2d.
    out_ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, groups=1)
    
    # They should not match because the custom kernel is hard coded for depthwise conv.
    # We assert that a significant difference is observed.
    diff = (out_custom - out_ref).abs().max().item()
    assert diff > 1e-3, f"Custom kernel output unexpectedly matched the reference output with diff={diff}"

# Test 2: Non-square kernel issue.
# Here we deliberately supply a rectangular (non-square) kernel.
# The custom kernel uses a single kernel_size obtained from weight.size(2) for both dimensions.
# This should cause an incorrect computation when weight.size(2) != weight.size(3).
def test_non_square_kernel_issue():
    device = "cuda"
    batch_size = 2
    in_channels = 3
    height = 20
    width = 20
    # Define a rectangular kernel (e.g., 3x2) instead of square.
    kernel_height = 3
    kernel_width = 2
    stride = 1
    padding = 1

    # For a depthwise convolution, PyTorch expects weight shape to be (in_channels, 1, kH, kW).
    weight_rect = torch.randn(in_channels, 1, kernel_height, kernel_width, device=device, dtype=torch.float32)
    bias = torch.randn(in_channels, device=device, dtype=torch.float32)
    x = torch.randn(batch_size, in_channels, height, width, device=device, dtype=torch.float32)
    
    # The custom kernel will use weight.size(2) (i.e. kernel_height) for both dimensions.
    mod = build_kernel()
    out_custom = mod.forward(x, weight_rect, bias, stride, padding, groups=in_channels)
    
    # Compute the reference using PyTorch's functional conv2d giving a proper rectangular kernel.
    out_ref = F.conv2d(x, weight_rect, bias, stride=stride, padding=padding, groups=in_channels)
    
    # They should not match because the custom kernel incorrectly treats the kernel as square.
    diff = (out_custom - out_ref).abs().max().item()
    assert diff > 1e-3, f"Custom kernel output unexpectedly matched the reference output with diff={diff}"
    
if __name__ == "__main__":
    pytest.main([__file__])
