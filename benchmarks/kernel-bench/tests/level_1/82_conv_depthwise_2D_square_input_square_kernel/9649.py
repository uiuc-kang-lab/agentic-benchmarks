
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Helper function to build and load the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test for "groups" parameter being ignored.
# Here we create a convolution where groups != in_channels.
# The PyTorch reference conv2d (with groups specified) will compute a different result
# compared to our kernel that assumes depthwise behavior.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_groups_parameter_ignored():
    # Create an input with in_channels = 4, but specify groups = 2.
    batch_size = 1
    in_channels = 4
    kernel_size = 3
    height, width = 16, 16
    stride = 1
    padding = 1
    groups = 2  # not equal to in_channels
    
    # Create input tensor and weight for convolution.
    # For standard conv2d with groups=2, weight shape should be (out_channels, in_channels/groups, kernel_h, kernel_w)
    # Here we set out_channels = in_channels for simplicity, but groups != in_channels.
    weight = torch.randn(in_channels, in_channels // groups, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    
    # Reference output using PyTorch conv2d with groups=2.
    ref_out = F.conv2d(x, weight, bias, stride=stride, padding=padding, groups=groups)
    
    # The custom kernel is designed for depthwise conv (groups == in_channels).
    # We pass groups to the kernel wrap but it will ignore it.
    kernel_module = build_kernel()
    # To mimic the depthwise setting expected by the kernel, weight is assumed to be of shape
    # (in_channels, 1, kernel_size, kernel_size). So we take the first channel slice for each group.
    # This deliberate mismatch should trigger the issue.
    weight_depthwise = weight.clone()  # not matching the intended assumption
    out = kernel_module.forward(x, weight_depthwise, bias, stride, padding, groups)
    torch.cuda.synchronize()
    
    # The outputs are expected to be different because the kernel ignores the groups argument.
    # We assert that the difference is significant.
    diff = (ref_out - out).abs().max().item()
    assert diff > 1e-3, f"Kernel unexpectedly produced an output matching the reference even with mismatched groups (diff: {diff})."

# Issue 2: Test for grid dimension exceeding CUDA limits.
# We create a situation where batch_size * in_channels is large.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_grid_dimension_limit():
    # Many channels and/or batch items to force batch_size*in_channels to exceed typical gridDim.z limit.
    # Note: The maximum gridDim.z is hardware dependent (often 65535).
    batch_size = 256
    in_channels = 256  # product = 65536, which is likely to exceed gridDim.z limits on many devices.
    kernel_size = 3
    height, width = 64, 64
    stride = 1
    padding = 1
    
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # For depthwise convolution expected weight shape is (in_channels, 1, kernel_size, kernel_size)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)
    
    kernel_module = build_kernel()
    # We expect a runtime error due to grid dimension limits.
    with pytest.raises(RuntimeError):
        out = kernel_module.forward(x, weight, bias, stride, padding, groups=in_channels)
        # Force CUDA synchronization to trigger any kernel launch issues.
        torch.cuda.synchronize()

# Issue 3: Test for tensor non-contiguity.
# We pass non-contiguous tensors to the kernel. The kernel assumes contiguous data for
# correct flat indexing, so the output should differ from the expected result.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input():
    batch_size = 2
    in_channels = 3
    kernel_size = 3
    height, width = 16, 16
    stride = 1
    padding = 1
    
    # Create a contiguous input and then make it non-contiguous by transposing spatial dimensions.
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    x_noncontig = x.transpose(2, 3)  # now non-contiguous
    # Force non-contiguity check
    assert not x_noncontig.is_contiguous(), "Test setup error: x_noncontig is unexpectedly contiguous."
    
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)
    
    # Reference: apply same non-contiguous input after making it contiguous for correct computation.
    ref_out = F.conv2d(x_noncontig.contiguous(), weight, bias, stride=stride, padding=padding)
    
    kernel_module = build_kernel()
    out = kernel_module.forward(x_noncontig, weight, bias, stride, padding, groups=in_channels)
    torch.cuda.synchronize()
    
    # Since the kernel does not account for non-contiguous layouts, the output will be incorrect.
    diff = (ref_out - out).abs().max().item()
    assert diff > 1e-3, f"Kernel output unexpectedly matches the reference on non-contiguous input (diff: {diff})."
