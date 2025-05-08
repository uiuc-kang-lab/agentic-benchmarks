
import torch
import torch.nn.functional as F
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper to run our custom CUDA conv forward function.
def custom_conv2d(x, weight, bias, stride, padding, dilation, groups):
    custom = build_kernel()
    if bias is None:
        bias_opt = torch.tensor([], device=x.device)
    else:
        bias_opt = bias
    return custom.forward(x, weight, bias_opt, stride, padding, dilation, groups)

# Test case 1: Kernel size mismatch.
# Here, we set kernel_size=5 but the CUDA kernel internally uses 3.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kernel_size_mismatch():
    batch_size = 1
    in_channels = 3
    out_channels = 8  # less than MIN_CHANNELS_THRESHOLD to trigger custom kernel.
    kernel_size = 5  # different than the hard-coded value 3.
    stride = 1
    padding = 2
    dilation = 1
    groups = 1
    # Ensure input dimensions are sufficiently large to not trigger fallback.
    height = 40  
    width = 40
    
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)
    # Create weight tensor for a 5x5 conv.
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    bias = None
    
    # Custom kernel call will still use 3 as kernel size.
    out_custom = custom_conv2d(x, weight, bias, stride, padding, dilation, groups)
    # Reference computation with F.conv2d.
    out_ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    
    # We expect a discrepancy due to kernel size conflict.
    assert not torch.allclose(out_custom, out_ref, atol=1e-4), "Custom kernel should fail for kernel size != 3"

# Test case 2: Dilation is not supported.
# Pass a dilation != 1 so that the expected output (using dilation) will differ.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dilation_ignored():
    batch_size = 1
    in_channels = 3
    out_channels = 8
    kernel_size = 3  # matches the hard-coded 3.
    stride = 1
    padding = 1
    dilation = 2  # non-default dilation; unsupported in custom kernel.
    groups = 1
    height = 40  
    width = 40
    
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    bias = None
    
    out_custom = custom_conv2d(x, weight, bias, stride, padding, dilation, groups)
    out_ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    
    # The difference should be significant because dilation is ignored in the custom kernel.
    assert not torch.allclose(out_custom, out_ref, atol=1e-4), "Custom kernel should fail when dilation != 1"

# Test case 3: Groups convolution is not supported.
# Use groups != 1 to test that the custom kernel does not compute grouped convolutions.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_groups_unsupported():
    batch_size = 1
    in_channels = 4
    groups = 2
    out_channels = 4   # Must be divisible by groups.
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    height = 40  
    width = 40
    
    # For grouped conv, the weight shape is (out_channels, in_channels/groups, kernel_size, kernel_size)
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    bias = None
    
    out_custom = custom_conv2d(x, weight, bias, stride, padding, dilation, groups)
    out_ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    assert not torch.allclose(out_custom, out_ref, atol=1e-4), "Custom kernel should fail for grouped convolution."

# Test case 4: Stride misalignment due to incorrect shared memory offset computation.
# Use a stride value != 1 (e.g., stride=2) which will amplify the misload in the input tile.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_stride_misalignment():
    batch_size = 1
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    stride = 2  # non-unit stride.
    padding = 1
    dilation = 1
    groups = 1
    height = 48  
    width = 48
    
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    bias = None
    
    out_custom = custom_conv2d(x, weight, bias, stride, padding, dilation, groups)
    out_ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    
    assert not torch.allclose(out_custom, out_ref, atol=1e-4), "Custom kernel should fail for stride != 1 due to misaligned input tile loading."
