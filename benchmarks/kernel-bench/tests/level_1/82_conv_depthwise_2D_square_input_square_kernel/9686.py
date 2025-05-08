
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to load the CUDA extension from "kernel.cu"
def build_kernel():
    cuda_module = load(
        name="depthwise_conv2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Reference function using PyTorch's own depthwise convolution
def reference_depthwise_conv2d(x, weight, bias, stride, padding):
    # groups equals in_channels for depthwise convolution.
    in_channels = x.size(1)
    conv = torch.nn.Conv2d(
        in_channels, in_channels, kernel_size=weight.shape[2],
        stride=stride, padding=padding, groups=in_channels, bias=bias is not None
    ).to(x.device).eval()
    with torch.no_grad():
        # Assign reference weight and bias
        conv.weight.copy_(weight)
        if bias is not None:
            conv.bias.copy_(bias)
    return conv(x)

# 1. Test: kernel_size > 5 (out-of-bound weights array indexing)
def test_kernel_size_too_large():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Create input and weight with kernel_size = 7, which exceeds the fixed-size storage of 25 elements.
    batch_size, in_channels, height, width = 2, 3, 32, 32
    kernel_size = 7
    stride = 1
    padding = 3  # so that output size equals input size
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)
    module = build_kernel()
    
    # This test is expected to trigger memory corruption in the kernel.
    with pytest.raises(Exception):
        out = module.forward(x, weight, bias, stride, padding, groups=in_channels)
        torch.cuda.synchronize()

# 2. Test: Non-depthwise groups parameter (groups != in_channels)
def test_incorrect_groups_param():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Even though our kernel implements depthwise conv, we try to call it with groups=1.
    batch_size, in_channels, height, width = 2, 4, 32, 32
    kernel_size = 3
    stride = 1
    padding = 1
    # When groups != in_channels the convolution should be different.
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)
    module = build_kernel()
    
    # The kernel ignores groups and always does a depthwise conv.
    # Compare with reference depthwise convolution (which requires groups==in_channels).
    out_custom = module.forward(x, weight, bias, stride, padding, groups=1)
    out_ref = reference_depthwise_conv2d(x, weight, bias, stride, padding)
    # They should not match because groups is mis-handled.
    with pytest.raises(AssertionError):
        assert torch.allclose(out_custom, out_ref, atol=1e-4), "Custom kernel output unexpectedly matches reference!"

# 3. Test: Grid dimension overflow (batch_size * in_channels exceeds gridDim.z limit)
def test_grid_dimension_overflow():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # CUDA grid dimension limit for z is typically 65535. Configure batch and channels so that:
    # batch_size * in_channels > 65535. We choose a configuration that forces a grid dimension overflow.
    # Note: This test may simply trigger a launch failure.
    batch_size = 200
    in_channels = 400  # product = 80,000 which exceeds typical grid z dimension limit.
    height = width = 16
    kernel_size = 3
    stride = 1
    padding = 1
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)
    module = build_kernel()
    
    with pytest.raises(Exception):
        out = module.forward(x, weight, bias, stride, padding, groups=in_channels)
        torch.cuda.synchronize()

# 4. Test: Noncontiguous input tensor causing potential misaligned memory accesses
def test_noncontiguous_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Create a contiguous tensor and then make a noncontiguous view using transpose.
    batch_size, in_channels, height, width = 2, 3, 32, 32
    kernel_size = 3
    stride = 1
    padding = 1
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Transpose to make a noncontiguous tensor and then transpose back along one dimension.
    x_noncontig = x.transpose(2, 3)
    # The kernel may assume contiguity; the result may be wrong.
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)
    module = build_kernel()
    
    out_custom = module.forward(x_noncontig.contiguous(), weight, bias, stride, padding, groups=in_channels)
    out_ref = reference_depthwise_conv2d(x_noncontig.contiguous(), weight, bias, stride, padding)
    # Force a failure if the input noncontiguity (or lack of checking) would yield a difference.
    with pytest.raises(AssertionError):
        assert torch.allclose(out_custom, out_ref, atol=1e-4), "Custom kernel output unexpectedly matches reference for noncontiguous input!"

