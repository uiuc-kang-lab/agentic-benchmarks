
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Function to build the CUDA kernel extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# ===================== Issue 1 Test =====================
# This test constructs a grouped convolution scenario where groups < in_channels.
# In a proper grouped convolution using PyTorch's F.conv2d, the output will be correct,
# but calling the CUDA kernel (which assumes depthwise convolution) will produce a wrong result.
def test_grouped_convolution_issue():
    # Use a grouped convolution with channels_per_group > 1.
    batch_size = 2
    in_channels = 4   # Total input channels.
    groups = 2        # groups = 2 => channels_per_group should be 2.
    kernel_h = 3
    kernel_w = 3
    stride = 1
    padding = 1

    # Create input tensor and weight tensor for a grouped convolution.
    # Standard conv2d weight shape: (out_channels, in_channels/groups, kernel_h, kernel_w).
    out_channels = in_channels  # To keep things simple.
    weight = torch.randn(out_channels, in_channels // groups, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    
    # Compute reference output using PyTorch's conv2d.
    ref_output = F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, dilation=1, groups=groups)

    # Prepare weight as assumed by our CUDA kernel.
    # Our kernel expects weight in shape: (groups, channels_per_group, kernel_h, kernel_w)
    # where out_channels = groups * channels_per_group.
    # In our grouped conv, weight shape is (out_channels, in_channels/groups, kernel_h, kernel_w).
    # For depthwise, in_channels/groups == 1, but here we have 2.
    # We supply the full weight tensor to the kernel despite this mismatch.
    cuda_module = build_kernel()
    
    # Invoke the kernel extension
    # The forward function in our kernel expects:
    #   x, weight, optional bias, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups
    output = cuda_module.forward(
        x, weight, bias,
        stride, stride,
        padding, padding,
        1, 1,  # dilation_h, dilation_w
        groups
    )

    # This test is designed to trigger the issue, so the CUDA kernel output is expected to be wrong.
    # We assert that the output of the CUDA kernel does NOT match the reference.
    assert not torch.allclose(output, ref_output, atol=1e-3), \
        "Kernel output unexpectedly matches reference output; expected mismatch due to wrong indexing for grouped convolution."

# ===================== Issue 2 Test =====================
# This test verifies that using a non-float32 tensor (e.g., float64) triggers an error.
# The CUDA kernel expects float32 pointers.
def test_data_type_issue():
    batch_size = 2
    in_channels = 3
    groups = in_channels  # depthwise scenario
    kernel_h = 3
    kernel_w = 5
    stride = 1
    padding = 0

    # Create input and weight tensors with dtype float64 instead of float32.
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float64)
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float64)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float64)

    cuda_module = build_kernel()

    with pytest.raises(RuntimeError):
        # The kernel code is hardcoded to use float pointers, and accessing data_ptr<float>()
        # on a double tensor should cause an error.
        cuda_module.forward(
            x, weight, bias,
            stride, stride,
            padding, padding,
            1, 1,  # dilation values
            groups
        )
